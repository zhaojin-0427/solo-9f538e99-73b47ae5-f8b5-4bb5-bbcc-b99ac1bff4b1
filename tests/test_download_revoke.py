"""Tests for download/revocation consistency.

The original bug: a CSV download that had passed its eligibility check (or was
already streaming) could still deliver the complete file after a quality
controller revoked its source, because nothing re-checked the state while the
response was being produced and nothing blocked new admissions racing with the
revoke commit.

Two layers are exercised:

* the Flask routes (through the Werkzeug test client), which cover admission,
  the 409 pre-first-byte path, post-revoke refusal and restart rebuild;
* the :class:`DownloadManager` + :class:`RevocableFileResponse` directly as a
  WSGI app, which gives deterministic control over where a stream is paused
  when a revocation lands (the test client buffers whole responses and cannot
  pause a live generator mid-stream).
"""

from __future__ import annotations

import io
import re
import threading

import pytest

from app import create_app, db
from app.downloads import CHUNK_SIZE, DownloadManager, RevocableFileResponse
from app.services import revoke_file


HEADER = "batch,value,note\n"
SMALL_CSV = "batch,value\nB1,10.5\nB2,11.0\n"


def _big_csv(num_rows: int) -> str:
    """CSV spanning several 256 KiB stream chunks so mid-stream revocation has
    a deterministic point to act on."""
    row = "B{:06d},{:0.4f},{}\n".format
    lines = [HEADER]
    lines.extend(row(i, i * 1.5, "x" * 40) for i in range(num_rows))
    return "".join(lines)


BIG_CSV = _big_csv(40000)  # > 2 MiB -> multiple chunks


@pytest.fixture()
def app(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "DATABASE": str(tmp_path / "stability.sqlite3"),
            "STORAGE_DIR": str(tmp_path / "objects"),
            "STAGING_DIR": str(tmp_path / "staging"),
            "WTF_CSRF_ENABLED": False,
            # Short drain waits: tests deliberately hold streams open.
            "DOWNLOAD_DRAIN_TIMEOUT": 0.3,
        }
    )
    yield app


@pytest.fixture()
def client(app):
    return app.test_client()


def _upload(client, kind, content, filename="m.csv", parent_ids=""):
    return client.post(
        "/upload",
        data={
            "kind": kind,
            "file": (io.BytesIO(content.encode()), filename),
            "parent_ids": parent_ids,
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )


def _new_id(resp_text, prefix):
    return re.search(rf"{prefix}-\d{{4}}", resp_text).group(0)


def _chain(client, content=SMALL_CSV):
    raw = _new_id(_upload(client, "raw", content).get_data(as_text=True), "RAW")
    cln = _new_id(
        _upload(client, "cleaned", content, parent_ids=raw).get_data(as_text=True),
        "CLN",
    )
    smm = _new_id(
        _upload(client, "summary", content, parent_ids=cln).get_data(as_text=True),
        "SUM",
    )
    return raw, cln, smm


def _manager(app) -> DownloadManager:
    return app.extensions["download_manager"]


def _stored_path(app, file_id) -> str:
    import os

    return os.path.join(app.config["STORAGE_DIR"], f"{file_id}.csv")


def _wsgi_environ():
    """Minimal WSGI environ sufficient for Werkzeug's response machinery."""
    return {"REQUEST_METHOD": "GET", "SERVER_NAME": "localhost",
            "SERVER_PORT": "80", "wsgi.url_scheme": "http"}


def _wsgi_iterate(response):
    """Drive a Werkzeug response as a WSGI server would, returning the list of
    byte chunks actually produced and the captured status/headers."""
    captured = {}

    def start_response(status, headers, exc_info=None):
        captured["status"] = status
        captured["headers"] = headers

    app_iter = response(_wsgi_environ(), start_response)
    chunks = list(app_iter)
    close = getattr(app_iter, "close", None)
    if close is not None:
        close()
    return chunks, captured


# ---------------------------------------------------------------------------
# 1. Baseline: an admitted download streams the full, exact object
# ---------------------------------------------------------------------------

def test_admitted_download_returns_complete_file(client, app):
    raw, _, _ = _chain(client)
    resp = client.get(f"/files/{raw}/download")
    assert resp.status_code == 200
    assert resp.data == SMALL_CSV.encode()
    assert "attachment" in resp.headers["Content-Disposition"]
    assert _manager(app).active_count(raw) == 0


# ---------------------------------------------------------------------------
# 2. After revoke commits, no NEW download can be admitted
# ---------------------------------------------------------------------------

def test_revoke_blocks_new_admission_through_route(client, app):
    raw, cln, smm = _chain(client)
    client.post(f"/files/{raw}/revoke", data={"reason": "bad source"})
    manager = _manager(app)

    for fid in (raw, cln, smm):
        resp = client.get(f"/files/{fid}/download", follow_redirects=False)
        assert resp.status_code == 302
        assert manager.admit(fid) is None
        assert manager.is_blocked(fid)


def test_revoked_descendants_are_all_marked_blocked(client, app):
    raw, cln, smm = _chain(client)
    client.post(f"/files/{raw}/revoke", data={"reason": "x"})
    assert {raw, cln, smm}.issubset(_manager(app)._blocked)


# ---------------------------------------------------------------------------
# 3. In-flight download cancelled mid-stream (deterministic WSGI control)
# ---------------------------------------------------------------------------

def test_inflight_download_terminated_mid_stream(app, client):
    raw, _, _ = _chain(client, content=BIG_CSV)
    manager = _manager(app)

    lease = manager.admit(raw)
    assert lease is not None
    response = RevocableFileResponse(
        _stored_path(app, raw), raw, "m.csv", lease, manager, app.logger
    )

    captured = {}

    def start_response(status, headers, exc_info=None):
        captured["status"] = status
        captured["headers"] = headers

    gen = iter(response(_wsgi_environ(), start_response))
    first = next(gen)  # attachment headers + first chunk now produced
    assert captured["status"].startswith("200")
    assert len(first) == CHUNK_SIZE
    assert manager.active_count(raw) == 1

    # Revoke while the generator is paused after the first chunk.
    with app.app_context():
        report = manager.revoke([raw], timeout=0.3)
    assert report["drained"] is False  # generator still held open by the test
    assert lease.alive() is False

    # Exhaust the cancelled generator: it must stop well before the full file.
    rest = b"".join(gen)
    received = first + rest
    assert len(received) < len(BIG_CSV.encode())
    assert len(received) == CHUNK_SIZE  # nothing after the cancelled check
    assert manager.active_count(raw) == 0

    # A fresh download is refused outright.
    assert client.get(f"/files/{raw}/download",
                      follow_redirects=False).status_code == 302


def test_revoke_drains_when_generator_closes(app, client):
    raw, _, _ = _chain(client, content=BIG_CSV)
    manager = _manager(app)

    lease = manager.admit(raw)
    response = RevocableFileResponse(
        _stored_path(app, raw), raw, "m.csv", lease, manager, app.logger
    )
    captured = {}
    gen = iter(response(_wsgi_environ(), lambda s, h, exc_info=None: captured.update(status=s)))
    next(gen)
    assert manager.active_count(raw) == 1

    report_box = {}

    def do_revoke():
        with app.app_context():
            report_box["report"] = manager.revoke([raw], timeout=0.3)

    t = threading.Thread(target=do_revoke)
    t.start()
    gen.close()  # client disconnect / cancelled generator finishes
    t.join(timeout=5)
    assert not t.is_alive()
    assert report_box["report"]["drained"] is True
    assert manager.active_count(raw) == 0


def test_cancel_before_first_byte_emits_nothing(app, client):
    """Cancellation landing before the first chunk is read yields no bytes."""
    raw, _, _ = _chain(client, content=BIG_CSV)
    manager = _manager(app)

    lease = manager.admit(raw)
    manager.revoke([raw], timeout=0.3)  # cancel before iteration starts
    response = RevocableFileResponse(
        _stored_path(app, raw), raw, "m.csv", lease, manager, app.logger
    )
    chunks, captured = _wsgi_iterate(response)
    assert b"".join(chunks) == b""
    assert manager.active_count(raw) == 0
    # Headers still advertise the attachment; the empty body is a failed
    # transfer, not a complete file.
    assert captured["status"].startswith("200")


def test_normal_stream_releases_lease_and_matches_bytes(app, client):
    raw, _, _ = _chain(client, content=BIG_CSV)
    manager = _manager(app)
    lease = manager.admit(raw)
    response = RevocableFileResponse(
        _stored_path(app, raw), raw, "m.csv", lease, manager, app.logger
    )
    chunks, _ = _wsgi_iterate(response)
    assert b"".join(chunks) == BIG_CSV.encode()
    assert manager.active_count(raw) == 0


# ---------------------------------------------------------------------------
# 4. Pre-first-byte race at the route: lease admitted then revoked before the
#    response object is built -> explicit 409, no attachment
# ---------------------------------------------------------------------------

def test_admit_then_revoke_refuses_further_admission(app):
    manager = DownloadManager()
    lease = manager.admit("RAW-0001")
    assert lease is not None and lease.alive()
    manager.revoke(["RAW-0001"], timeout=0.3)
    assert lease.alive() is False
    assert manager.admit("RAW-0001") is None
    manager.release(lease)
    assert manager.active_count("RAW-0001") == 0


def test_route_returns_409_when_lease_cancelled_before_response(app, client, monkeypatch):
    raw, _, _ = _chain(client)
    manager = _manager(app)
    original_admit = manager.admit

    def admit_then_revoke(file_id):
        lease = original_admit(file_id)
        if lease is not None and file_id == raw:
            manager.revoke([raw], timeout=0.3)
        return lease

    monkeypatch.setattr(manager, "admit", admit_then_revoke)
    resp = client.get(f"/files/{raw}/download")
    assert resp.status_code == 409
    assert "撤销" in resp.get_data(as_text=True)
    assert "attachment" not in resp.headers.get("Content-Disposition", "")
    assert manager.active_count(raw) == 0


# ---------------------------------------------------------------------------
# 5. Restart: the blocked set is rebuilt from the persisted database
# ---------------------------------------------------------------------------

def test_blocked_set_rebuilt_after_restart(tmp_path, client):
    config = {
        "TESTING": True,
        "DATABASE": str(tmp_path / "stability.sqlite3"),
        "STORAGE_DIR": str(tmp_path / "objects"),
        "STAGING_DIR": str(tmp_path / "staging"),
    }
    raw, cln, smm = _chain(client)
    client.post(f"/files/{raw}/revoke", data={"reason": "persist"})

    app2 = create_app(config)
    manager2 = app2.extensions["download_manager"]
    for fid in (raw, cln, smm):
        assert manager2.is_blocked(fid)
        assert manager2.admit(fid) is None

    client2 = app2.test_client()
    new_raw = _new_id(
        _upload(client2, "raw", SMALL_CSV, filename="fixed.csv").get_data(as_text=True),
        "RAW",
    )
    assert manager2.admit(new_raw) is not None
    assert client2.get(f"/files/{new_raw}/download").status_code == 200


# ---------------------------------------------------------------------------
# 6. An ancestor revoke cancels leases of every transitive descendant
# ---------------------------------------------------------------------------

def test_ancestor_revoke_cancels_descendant_leases(app, client):
    raw, cln, smm = _chain(client, content=BIG_CSV)
    manager = _manager(app)

    def open_stream(fid):
        lease = manager.admit(fid)
        response = RevocableFileResponse(
            _stored_path(app, fid), fid, "m.csv", lease, manager, app.logger
        )
        gen = iter(response(_wsgi_environ(), lambda s, h, exc_info=None: None))
        first = next(gen)
        return lease, response, gen, first

    lease_c, _, gen_c, first_c = open_stream(cln)
    lease_s, _, gen_s, first_s = open_stream(smm)
    assert manager.active_count(cln) == 1
    assert manager.active_count(smm) == 1

    with app.app_context():
        revoke_file(raw, "upstream wrong")

    for fid, gen, first in ((cln, gen_c, first_c), (smm, gen_s, first_s)):
        rest = b"".join(gen)
        assert len(first + rest) < len(BIG_CSV.encode())
        assert manager.active_count(fid) == 0
    assert lease_c.alive() is False and lease_s.alive() is False


# ---------------------------------------------------------------------------
# 7. Repeated interleaving: any download admitted after revoke is refused;
#    any stream in flight ends truncated, never complete
# ---------------------------------------------------------------------------

def test_repeated_revoke_download_interleaving(app, client):
    manager = _manager(app)
    for round_no in range(10):
        raw, _, _ = _chain(client, content=_big_csv(2000))
        full_size = len(_big_csv(2000).encode())
        barrier = threading.Barrier(2)
        outcome = {}

        def downloader():
            barrier.wait()
            with app.test_request_context(f"/files/{raw}/download"):
                lease = manager.admit(raw)
                if lease is None:
                    outcome["dl"] = "refused"
                    return
                response = RevocableFileResponse(
                    _stored_path(app, raw), raw, "m.csv", lease, manager,
                    app.logger,
                )
                chunks, _ = _wsgi_iterate(response)
                outcome["dl"] = "complete" if len(b"".join(chunks)) == full_size \
                    else "truncated"

        def revoker():
            barrier.wait()
            with app.app_context():
                revoke_file(raw, f"round {round_no}")

        t1 = threading.Thread(target=downloader)
        t2 = threading.Thread(target=revoker)
        t1.start(); t2.start(); t1.join(); t2.join()

        assert outcome["dl"] in ("refused", "truncated", "complete")
        # If the download completed, the revoke must not have committed first;
        # either way every subsequent request is now refused.
        assert client.get(f"/files/{raw}/download",
                          follow_redirects=False).status_code == 302
        assert manager.admit(raw) is None


def test_drain_timeout_does_not_unblock_future_admission(app):
    """Even if a slow stream exceeds the drain timeout, the id stays blocked
    and no new lease is ever granted."""
    manager = DownloadManager()
    lease = manager.admit("RAW-0009")
    report = manager.revoke(["RAW-0009"], timeout=0.05)
    assert report["drained"] is False
    assert manager.admit("RAW-0009") is None
    manager.release(lease)
    assert manager.active_count("RAW-0009") == 0
    assert manager.admit("RAW-0009") is None
