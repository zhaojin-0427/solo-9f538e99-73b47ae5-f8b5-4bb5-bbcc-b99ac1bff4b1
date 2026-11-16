"""Flask application factory and HTTP routes."""

from __future__ import annotations

import os

from flask import (
    Flask,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)

from . import db, downloads, services


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__)
    instance = os.environ.get(
        "DATA_DIR", os.path.join(os.getcwd(), "data")
    )
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-only-change-me"),
        DATABASE=os.path.join(instance, "stability.sqlite3"),
        STORAGE_DIR=os.path.join(instance, "objects"),
        STAGING_DIR=os.path.join(instance, "staging"),
        MAX_CONTENT_LENGTH=64 * 1024 * 1024,
        # How long a revocation waits for already-open downloads to settle.
        # Kept below the gunicorn worker timeout; overridable for tuning/tests.
        DOWNLOAD_DRAIN_TIMEOUT=float(
            os.environ.get("DOWNLOAD_DRAIN_TIMEOUT", downloads.DEFAULT_DRAIN_TIMEOUT)
        ),
    )
    if config:
        app.config.update(config)

    db.init_app(app)
    with app.app_context():
        services.sweep_working_dirs()
        # In-process download gate.  Its blocked set is derived from the
        # persisted database on every (re)start, so eligibility after restart
        # stays identical while the in-flight race window can be closed.
        manager = downloads.DownloadManager()
        app.extensions["download_manager"] = manager
        conn = db.get_db()
        manager.rebuild_blocked(
            set(db.blocked_by_map(conn))
            | {
                row["id"]
                for row in conn.execute(
                    "SELECT id FROM files WHERE revoked = 1"
                )
            }
        )

    # -- pages -------------------------------------------------------------

    @app.route("/")
    def index():
        conn = db.get_db()
        files = db.list_all_files(conn)
        blocked = db.blocked_by_map(conn)
        rows = []
        for row in files:
            rows.append(
                {
                    "row": row,
                    "available": (not row["revoked"])
                    and row["id"] not in blocked,
                    "blocked_by": blocked.get(row["id"]),
                }
            )
        events = db.recent_events(conn)
        return render_template(
            "index.html",
            rows=rows,
            events=events,
            kinds=services.KINDS,
            kind_labels=services.KIND_LABELS,
        )

    @app.route("/files/<file_id>")
    def file_detail(file_id):
        conn = db.get_db()
        row = db.get_file(conn, file_id)
        if row is None:
            abort(404)
        blocker, blocker_name, blocker_path = db.find_blocking_ancestor(
            conn, file_id
        )
        available = (not row["revoked"]) and blocker is None
        parents = db.list_parents(conn, file_id)
        blocked_map = db.blocked_by_map(conn)
        blocked_parent_ids = {
            p["id"] for p in parents if p["id"] in blocked_map
        }
        children = db.list_children(conn, file_id)
        blocked_child_ids = {
            c["id"] for c in children if c["id"] in blocked_map
        }
        chains = [
            dict(r)
            for r in db.descendant_chains(conn, file_id)
            if r["descendant_id"] != file_id
        ]
        affected = sorted({r["descendant_id"] for r in chains})
        return render_template(
            "detail.html",
            row=row,
            available=available,
            blocker=blocker,
            blocker_name=blocker_name,
            blocker_path=blocker_path,
            parents=parents,
            children=children,
            blocked_parent_ids=blocked_parent_ids,
            blocked_child_ids=blocked_child_ids,
            chains=chains,
            affected=affected,
            kind_labels=services.KIND_LABELS,
        )

    # -- actions -----------------------------------------------------------

    @app.route("/upload", methods=["POST"])
    def upload():
        try:
            file_id = services.register_file(
                request.files.get("file"),
                request.form.get("kind", ""),
                request.form.get("parent_ids", ""),
            )
        except services.ServiceError as exc:
            flash(f"上传被拒绝：{exc}", "error")
            return redirect(url_for("index"))
        flash(f"登记成功，文件 ID 为 {file_id}", "success")
        return redirect(url_for("file_detail", file_id=file_id))

    @app.route("/files/<file_id>/revoke", methods=["POST"])
    def revoke(file_id):
        try:
            result = services.revoke_file(
                file_id, request.form.get("reason", "")
            )
        except services.ServiceError as exc:
            flash(f"撤销失败：{exc}", "error")
            return redirect(url_for("file_detail", file_id=file_id))
        if result["already_revoked"]:
            flash(f"文件 {file_id} 此前已被撤销", "notice")
        else:
            flash(
                f"文件 {file_id} 已撤销；受影响的派生文件 "
                f"{len(result['affected'])} 个："
                + (", ".join(result["affected"]) or "无"),
                "success",
            )
        return redirect(url_for("file_detail", file_id=file_id))

    @app.route("/files/<file_id>/download")
    def download(file_id):
        conn = db.get_db()
        row = db.get_file(conn, file_id)
        if row is None:
            abort(404)
        if row["revoked"]:
            flash(f"文件 {file_id} 已被撤销，禁止下载", "error")
            return redirect(url_for("file_detail", file_id=file_id))
        blocker, blocker_name, _ = db.find_blocking_ancestor(conn, file_id)
        if blocker is not None:
            flash(
                f"文件 {file_id} 的上游 {blocker}（{blocker_name}）"
                "已被撤销，本文件禁止下载",
                "error",
            )
            return redirect(url_for("file_detail", file_id=file_id))
        path = os.path.join(
            current_app.config["STORAGE_DIR"], row["stored_name"]
        )
        if not os.path.isfile(path):
            # Should never happen: registration is atomic.  Fail closed.
            flash("存储对象缺失，为安全起见禁止下载，请联系管理员", "error")
            return redirect(url_for("file_detail", file_id=file_id))

        # Admission under the manager lock serialises against revocation:
        # once a revoke has marked this id blocked, no new lease is granted.
        manager = app.extensions["download_manager"]
        lease = manager.admit(file_id)
        if lease is None:
            flash(
                f"文件 {file_id} 已撤销或被依赖阻断，下载请求未被放行",
                "error",
            )
            return redirect(url_for("file_detail", file_id=file_id))
        # Re-check before producing the attachment: a revoke committing in the
        # admission-to-response window is answered with an explicit 409 while
        # no byte has been promised; the response generator keeps checking per
        # chunk for revocations landing after streaming has begun.
        if not lease.alive():
            manager.release(lease)
            return (
                f"文件 {file_id} 在下载开始前已被撤销，下载终止。\n",
                downloads.REVOKED_HTTP_STATUS,
                {"Content-Type": "text/plain; charset=utf-8"},
            )
        return downloads.RevocableFileResponse(
            path=path,
            file_id=file_id,
            download_name=row["filename"],
            lease=lease,
            manager=manager,
            logger=app.logger,
        )

    @app.errorhandler(413)
    def too_large(_err):
        flash("上传文件超过 64MB 限制", "error")
        return redirect(url_for("index"))

    @app.errorhandler(404)
    def not_found(_err):
        return render_template("404.html"), 404

    return app
