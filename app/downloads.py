"""Download admission gate and in-flight lease management.

The eligibility check in a view and the actual byte streaming of a response do
not happen atomically: ``send_file`` opens and reads the object only after the
view has returned, and a large file is streamed chunk by chunk.  A revocation
committed in that window used to go unnoticed, so a download could still
deliver the complete file after the source had been revoked.

This module closes the window with an in-process :class:`DownloadManager`:

* ``admit`` hands out a lease while holding the manager lock.  Revocation marks
  the revoked file and every transitive descendant as blocked under the same
  lock, after which ``admit`` refuses every new request.  Admission and
  revocation are therefore totally ordered with respect to each other.
* Every streaming chunk re-checks the lease.  A lease whose file becomes
  revoked is cancelled, so the stream is terminated promptly instead of
  leaking the whole object.  If the cancellation happens before the response
  is built the route answers an explicit ``409`` (no byte promised); once the
  attachment headers have been emitted the generator stops early, which the
  client observes as an incomplete, failed download rather than a saved full
  file.
* ``revoke`` waits for the affected in-flight leases to drain, so once the
  revocation request has returned no download of an affected file is still
  open.

The manager is intentionally in-process state.  The deployment runs a single
gunicorn worker (threads share the manager), which matches the SQLite
single-writer model used by the write transactions.  Eligibility remains fully
derivable from the persisted database; the in-memory blocked set is rebuilt
from the database on every startup and is a race-closing cache, never the
source of truth.
"""

from __future__ import annotations

import os
import threading
import time

from werkzeug.wrappers import Response

# Streamed block size.  Smaller chunks mean fewer bytes can leave the process
# between a revocation and the next lease check; 256 KiB bounds that window.
CHUNK_SIZE = 256 * 1024

# Maximum time a revocation waits for already-open downloads to settle.  It is
# below the gunicorn worker timeout so the revoke request itself never stalls
# the worker past its own limit.
DEFAULT_DRAIN_TIMEOUT = 15.0


class _Lease:
    """A single admitted, potentially in-flight download."""

    __slots__ = ("file_id", "cancelled", "bytes_sent")

    def __init__(self, file_id: str):
        self.file_id = file_id
        self.cancelled = False
        self.bytes_sent = 0

    def alive(self) -> bool:
        return not self.cancelled


class DownloadManager:
    """Process-wide lock, blocked-id set and active-lease registry."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._blocked: set[str] = set()
        self._active: dict[str, set[_Lease]] = {}

    # -- startup / restart --------------------------------------------------

    def rebuild_blocked(self, blocked_ids) -> None:
        """Replace the in-memory blocked set from persisted database state.

        Called at application startup (including after a process restart), so
        files revoked in a previous life are refused admission immediately.
        """
        with self._cond:
            self._blocked = set(blocked_ids)

    # -- admission ----------------------------------------------------------

    def admit(self, file_id: str) -> _Lease | None:
        """Grant a download lease, or ``None`` when the file is blocked.

        Runs under the same lock as :meth:`revoke`, so a revocation that has
        marked the file blocked can never race with a successful admission.
        """
        with self._cond:
            if file_id in self._blocked:
                return None
            lease = _Lease(file_id)
            self._active.setdefault(file_id, set()).add(lease)
            return lease

    def release(self, lease: _Lease) -> None:
        """Return an admitted lease, regardless of how the stream ended."""
        with self._cond:
            bucket = self._active.get(lease.file_id)
            if bucket is not None:
                bucket.discard(lease)
                if not bucket:
                    del self._active[lease.file_id]
                self._cond.notify_all()

    def is_blocked(self, file_id: str) -> bool:
        with self._cond:
            return file_id in self._blocked

    def active_count(self, file_id: str) -> int:
        with self._cond:
            return len(self._active.get(file_id, ()))

    # -- revocation ---------------------------------------------------------

    def revoke(
        self, file_ids, *, timeout: float = DEFAULT_DRAIN_TIMEOUT
    ) -> dict:
        """Block ``file_ids`` (the revoked file plus its descendants), cancel
        every lease touching them and wait for the leases to drain.

        Returns a small report describing whether all in-flight downloads had
        settled before ``timeout``.  The blocked set is updated either way, so
        new admissions are refused even if a slow client kept a stream open.
        """
        targets = set(file_ids)
        with self._cond:
            self._blocked.update(targets)
            for file_id in targets:
                for lease in self._active.get(file_id, ()):
                    lease.cancelled = True

            deadline = time.monotonic() + timeout
            while any(self._active.get(file_id) for file_id in targets):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    stuck = {
                        file_id: len(self._active.get(file_id, ()))
                        for file_id in targets
                        if self._active.get(file_id)
                    }
                    return {"drained": False, "stuck": stuck, "blocked": sorted(targets)}
                self._cond.wait(remaining)
            return {"drained": True, "stuck": {}, "blocked": sorted(targets)}


def get_download_manager():
    """Fetch the manager registered on the current Flask application."""
    from flask import current_app

    return current_app.extensions["download_manager"]


# HTTP code used by the download route when a lease is refused/revoked before
# the response is produced: the download never started, so answer 409 rather
# than redirecting an attachment request to an HTML page.
REVOKED_HTTP_STATUS = 409


class RevocableFileResponse(Response):
    """Streams a stored object while honouring a live download lease.

    The route owns the pre-first-byte decision: it admits the lease and
    re-checks it before returning this response, so a revocation landing that
    early produces a clean ``409`` while no byte has been promised.  Once this
    response is iterated the attachment headers have already been emitted, so
    the generator's own lease check handles a revocation landing mid-stream by
    terminating early: the client observes a failed, truncated transfer rather
    than a complete file of a revoked source.

    The lease is always released -- including on client disconnects and I/O
    errors -- so revocation draining cannot hang on a dead download.
    """

    def __init__(self, path: str, file_id: str, download_name: str,
                 lease: _Lease, manager: "DownloadManager", logger=None):
        super().__init__()
        self.file_path = path
        self.file_id = file_id
        self.lease = lease
        self.manager = manager
        self.logger = logger
        self.mimetype = "text/csv"
        try:
            self.content_length = os.path.getsize(path)
        except OSError:
            self.content_length = None
        filename = os.path.basename(download_name or f"{file_id}.csv")
        self.headers.add(
            "Content-Disposition", "attachment", filename=filename
        )
        # Standard Werkzeug pattern (the same one send_file uses): the
        # response body is a generator; get_app_iter/iter_encoded then consume
        # it both in real WSGI servers and in the test client.
        self.response = self._stream()

    def _stream(self):
        lease = self.lease
        manager = self.manager
        path = self.file_path
        file_id = self.file_id
        logger = self.logger

        # Guard for the tiny window between returning this response and the
        # first iteration: emit nothing rather than any byte of a now-revoked
        # file.  Headers already promise an attachment, so an empty body reads
        # to the client as a failed/truncated transfer, never a complete file.
        if not lease.alive():
            manager.release(lease)
            return
        try:
            fh = open(path, "rb")
        except OSError:
            if logger is not None:
                logger.exception(
                    "storage open failed for download %s", file_id
                )
            manager.release(lease)
            return

        try:
            while True:
                if not lease.alive():
                    # Attachment headers already left with the first byte.
                    # Stop without producing the rest of the file.
                    if logger is not None:
                        logger.warning(
                            "download %s terminated mid-stream after "
                            "revocation; %d bytes had been sent",
                            file_id, lease.bytes_sent,
                        )
                    return
                chunk = fh.read(CHUNK_SIZE)
                if not chunk:
                    return
                lease.bytes_sent += len(chunk)
                yield chunk
        except OSError:
            if logger is not None:
                logger.exception(
                    "storage read failed for download %s", file_id
                )
        finally:
            fh.close()
            manager.release(lease)
