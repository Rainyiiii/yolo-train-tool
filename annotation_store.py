# -*- coding: utf-8 -*-
"""SQLite-backed collaboration state for the YOLO Team annotation server."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import sqlite3
import time
import uuid
from io import BytesIO
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ROLES = {"admin", "reviewer", "annotator"}
STATUSES = {"unassigned", "assigned", "in_progress", "submitted", "approved", "rejected"}
USERNAME_RE = re.compile(r"^[\w\-\u4e00-\u9fff]{2,32}$", re.UNICODE)
PASSWORD_ITERATIONS = 310_000
SESSION_SECONDS = 7 * 24 * 60 * 60
LOCK_SECONDS = 5 * 60


class AnnotationError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class ClosingConnection(sqlite3.Connection):
    """Commit/rollback like sqlite3.Connection, then always release Windows file handles."""

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        try:
            return bool(super().__exit__(exc_type, exc_value, traceback))
        finally:
            self.close()


def _now() -> int:
    return int(time.time())


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _parse_json(value: str | None, default: Any) -> Any:
    try:
        return json.loads(value or "")
    except (TypeError, ValueError):
        return default


def hash_password(password: str) -> str:
    if len(password) < 8:
        raise AnnotationError("密码至少需要 8 个字符。")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        PASSWORD_ITERATIONS,
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt_text, digest_text = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_text.encode("ascii"))
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


class AnnotationStore:
    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).expanduser().resolve()
        self.db_path = self.workspace / "annotation_hub.sqlite3"
        self.projects_dir = self.workspace / "projects"
        self.exports_dir = self.workspace / "exports"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=20, factory=ClosingConnection)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at INTEGER NOT NULL,
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    labels_json TEXT NOT NULL,
                    task_type TEXT NOT NULL DEFAULT 'detect',
                    platform_project_id TEXT,
                    source_root TEXT NOT NULL DEFAULT '',
                    review_enabled INTEGER NOT NULL DEFAULT 0,
                    created_by INTEGER NOT NULL REFERENCES users(id),
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    original_name TEXT NOT NULL,
                    relative_source TEXT NOT NULL,
                    stored_name TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    assignee_id INTEGER REFERENCES users(id),
                    status TEXT NOT NULL DEFAULT 'unassigned',
                    annotations_json TEXT NOT NULL DEFAULT '[]',
                    revision INTEGER NOT NULL DEFAULT 0,
                    lock_user_id INTEGER REFERENCES users(id),
                    lock_expires_at INTEGER,
                    review_comment TEXT NOT NULL DEFAULT '',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    UNIQUE(project_id, relative_source)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
                    item_id INTEGER REFERENCES items(id) ON DELETE CASCADE,
                    user_id INTEGER REFERENCES users(id),
                    action TEXT NOT NULL,
                    detail_json TEXT NOT NULL DEFAULT '{}',
                    created_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_items_project_status ON items(project_id, status);
                CREATE INDEX IF NOT EXISTS idx_items_assignee ON items(assignee_id, status);
                CREATE INDEX IF NOT EXISTS idx_sessions_expiry ON sessions(expires_at);
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(projects)").fetchall()}
            if "platform_project_id" not in columns:
                connection.execute("ALTER TABLE projects ADD COLUMN platform_project_id TEXT")
            if "source_root" not in columns:
                connection.execute("ALTER TABLE projects ADD COLUMN source_root TEXT NOT NULL DEFAULT ''")
            if "review_enabled" not in columns:
                connection.execute("ALTER TABLE projects ADD COLUMN review_enabled INTEGER NOT NULL DEFAULT 0")
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_projects_platform_project_id "
                "ON projects(platform_project_id) WHERE platform_project_id IS NOT NULL"
            )

    def needs_setup(self) -> bool:
        with self.connect() as connection:
            return connection.execute("SELECT 1 FROM users LIMIT 1").fetchone() is None

    def _validate_username(self, username: str) -> str:
        value = str(username or "").strip()
        if not USERNAME_RE.fullmatch(value):
            raise AnnotationError("用户名需要 2–32 个字符，只能使用中文、字母、数字、下划线或连字符。")
        return value

    def create_user(self, username: str, password: str, role: str, actor: dict[str, Any] | None = None) -> dict[str, Any]:
        username = self._validate_username(username)
        if role not in ROLES:
            raise AnnotationError("用户角色无效。")
        if actor is None:
            if not self.needs_setup() or role != "admin":
                raise AnnotationError("只有首次初始化可以直接创建管理员。", 403)
        elif actor["role"] != "admin":
            raise AnnotationError("只有管理员可以创建用户。", 403)
        encoded = hash_password(password)
        try:
            with self.connect() as connection:
                cursor = connection.execute(
                    "INSERT INTO users(username,password_hash,role,created_at) VALUES(?,?,?,?)",
                    (username, encoded, role, _now()),
                )
                user_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise AnnotationError("用户名已经存在。") from exc
        return {"id": user_id, "username": username, "role": role, "active": True}

    def authenticate(self, username: str, password: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id,username,password_hash,role,active FROM users WHERE username=? COLLATE NOCASE",
                (str(username or "").strip(),),
            ).fetchone()
        if row is None or not row["active"] or not verify_password(password, row["password_hash"]):
            raise AnnotationError("用户名或密码不正确。", 401)
        return {"id": row["id"], "username": row["username"], "role": row["role"], "active": bool(row["active"])}

    def create_session(self, user_id: int) -> str:
        token = secrets.token_urlsafe(36)
        token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
        now = _now()
        with self.connect() as connection:
            connection.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
            connection.execute(
                "INSERT INTO sessions(token_hash,user_id,expires_at,created_at) VALUES(?,?,?,?)",
                (token_hash, user_id, now + SESSION_SECONDS, now),
            )
        return token

    def session_user(self, token: str) -> dict[str, Any] | None:
        if not token:
            return None
        token_hash = hashlib.sha256(token.encode("ascii", errors="ignore")).hexdigest()
        now = _now()
        with self.connect() as connection:
            row = connection.execute(
                """SELECT u.id,u.username,u.role,u.active FROM sessions s
                   JOIN users u ON u.id=s.user_id
                   WHERE s.token_hash=? AND s.expires_at>?""",
                (token_hash, now),
            ).fetchone()
        if row is None or not row["active"]:
            return None
        return {"id": row["id"], "username": row["username"], "role": row["role"], "active": True}

    def delete_session(self, token: str) -> None:
        if not token:
            return
        token_hash = hashlib.sha256(token.encode("ascii", errors="ignore")).hexdigest()
        with self.connect() as connection:
            connection.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash,))

    def list_users(self, actor: dict[str, Any]) -> list[dict[str, Any]]:
        if actor["role"] not in {"admin", "reviewer"}:
            raise AnnotationError("没有查看成员列表的权限。", 403)
        with self.connect() as connection:
            rows = connection.execute("SELECT id,username,role,active,created_at FROM users ORDER BY username COLLATE NOCASE").fetchall()
        return [{**dict(row), "active": bool(row["active"])} for row in rows]

    def create_project(
        self,
        actor: dict[str, Any],
        name: str,
        labels: Iterable[str],
        source_dir: str | Path | None = None,
        *,
        task_type: str = "detect",
        platform_project_id: str = "",
        review_enabled: bool = False,
    ) -> dict[str, Any]:
        if actor["role"] not in {"admin", "reviewer"}:
            raise AnnotationError("只有管理员或审核员可以创建项目。", 403)
        project_name = str(name or "").strip()
        if not project_name or len(project_name) > 80:
            raise AnnotationError("项目名称不能为空且不能超过 80 个字符。")
        clean_labels: list[str] = []
        for label in labels:
            value = str(label).strip()
            if value and value not in clean_labels:
                clean_labels.append(value)
        if not clean_labels:
            raise AnnotationError("至少需要一个类别。")
        if task_type != "detect":
            raise AnnotationError("当前标注中心仅支持目标检测项目。")
        source_root = str(Path(source_dir).expanduser().resolve()) if source_dir else ""
        now = _now()
        with self.connect() as connection:
            cursor = connection.execute(
                """INSERT INTO projects(
                       name,labels_json,task_type,platform_project_id,source_root,review_enabled,
                       created_by,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    project_name, _json(clean_labels), task_type, platform_project_id or None,
                    source_root, int(review_enabled), actor["id"], now, now,
                ),
            )
            project_id = int(cursor.lastrowid)
            self._event(connection, project_id, None, actor["id"], "project_created", {"name": project_name})
        imported = self.import_images(actor, project_id, source_dir) if source_dir else 0
        return {
            "id": project_id,
            "name": project_name,
            "labels": clean_labels,
            "imported": imported,
            "platform_project_id": platform_project_id,
            "review_enabled": bool(review_enabled),
        }

    def import_images(
        self,
        actor: dict[str, Any],
        project_id: int,
        source_dir: str | Path,
        *,
        allow_empty: bool = False,
    ) -> int:
        if actor["role"] not in {"admin", "reviewer"}:
            raise AnnotationError("只有管理员或审核员可以导入图片。", 403)
        source = Path(source_dir).expanduser().resolve()
        if not source.is_dir():
            raise AnnotationError("图片来源必须是服务端可访问的有效文件夹。")
        project = self.get_project(project_id, actor)
        destination = self.projects_dir / str(project_id) / "images"
        destination.mkdir(parents=True, exist_ok=True)
        candidates = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if not candidates:
            if allow_empty:
                return 0
            raise AnnotationError("来源文件夹中没有找到支持的图片。")
        if len(candidates) > 100_000:
            raise AnnotationError("单次导入最多支持 100000 张图片，请拆分项目。")
        imported = 0
        now = _now()
        with self.connect() as connection:
            for image_path in candidates:
                relative_source = image_path.relative_to(source).as_posix()
                if connection.execute(
                    "SELECT 1 FROM items WHERE project_id=? AND relative_source=?",
                    (project_id, relative_source),
                ).fetchone():
                    continue
                try:
                    with Image.open(image_path) as image:
                        width, height = image.size
                        image.verify()
                except (OSError, ValueError):
                    continue
                stored_name = f"{uuid.uuid4().hex}{image_path.suffix.lower()}"
                shutil.copy2(image_path, destination / stored_name)
                connection.execute(
                    """INSERT INTO items(
                           project_id,original_name,relative_source,stored_name,width,height,created_at,updated_at
                       ) VALUES(?,?,?,?,?,?,?,?)""",
                    (project_id, image_path.name, relative_source, stored_name, width, height, now, now),
                )
                imported += 1
            connection.execute("UPDATE projects SET updated_at=? WHERE id=?", (now, project_id))
            self._event(connection, project_id, None, actor["id"], "images_imported", {"count": imported, "source": str(source)})
        return imported

    def sync_platform_projects(self, actor: dict[str, Any], platform_projects: Iterable[dict[str, Any]]) -> dict[str, int]:
        """Create/update annotation projects from the platform registry without duplicating them."""
        if actor["role"] not in {"admin", "reviewer"}:
            return {"created": 0, "updated": 0, "imported": 0, "skipped": 0}
        summary = {"created": 0, "updated": 0, "imported": 0, "skipped": 0}
        for raw in platform_projects:
            platform_id = str(raw.get("id") or "").strip()
            name = str(raw.get("name") or "").strip()
            task_type = str(raw.get("task") or "detect").strip().lower()
            if not platform_id or not name or task_type != "detect":
                summary["skipped"] += 1
                continue
            labels = list(dict.fromkeys(str(value).strip() for value in raw.get("labels", []) if str(value).strip())) or ["object"]
            source_text = str(raw.get("dataset_root") or raw.get("root") or "").strip()
            source = Path(source_text).expanduser().resolve() if source_text else None
            now = _now()
            with self.connect() as connection:
                row = connection.execute(
                    "SELECT id FROM projects WHERE platform_project_id=?",
                    (platform_id,),
                ).fetchone()
                if row is None:
                    legacy = connection.execute(
                        "SELECT id FROM projects WHERE platform_project_id IS NULL AND name=? ORDER BY id LIMIT 2",
                        (name,),
                    ).fetchall()
                    row = legacy[0] if len(legacy) == 1 else None
                if row is None:
                    cursor = connection.execute(
                        """INSERT INTO projects(
                               name,labels_json,task_type,platform_project_id,source_root,review_enabled,
                               created_by,created_at,updated_at
                           ) VALUES(?,?,?,?,?,0,?,?,?)""",
                        (name, _json(labels), task_type, platform_id, str(source or ""), actor["id"], now, now),
                    )
                    annotation_project_id = int(cursor.lastrowid)
                    self._event(connection, annotation_project_id, None, actor["id"], "platform_project_synced", {"platform_project_id": platform_id})
                    summary["created"] += 1
                else:
                    annotation_project_id = int(row["id"])
                    connection.execute(
                        """UPDATE projects SET name=?,labels_json=?,task_type=?,platform_project_id=?,
                           source_root=?,updated_at=? WHERE id=?""",
                        (name, _json(labels), task_type, platform_id, str(source or ""), now, annotation_project_id),
                    )
                    summary["updated"] += 1
            if source and source.is_dir():
                summary["imported"] += self.import_images(actor, annotation_project_id, source, allow_empty=True)
        return summary

    def set_project_review_mode(self, actor: dict[str, Any], project_id: int, enabled: bool) -> dict[str, Any]:
        if actor["role"] not in {"admin", "reviewer"}:
            raise AnnotationError("只有管理员或审核员可以修改审核模式。", 403)
        self.get_project(project_id, actor)
        with self.connect() as connection:
            connection.execute(
                "UPDATE projects SET review_enabled=?,updated_at=? WHERE id=?",
                (int(enabled), _now(), project_id),
            )
            if not enabled:
                connection.execute(
                    """UPDATE items SET status='approved',lock_user_id=NULL,lock_expires_at=NULL,
                       updated_at=? WHERE project_id=? AND status='submitted'""",
                    (_now(), project_id),
                )
            self._event(connection, project_id, None, actor["id"], "review_mode_changed", {"enabled": bool(enabled)})
        return self.get_project(project_id, actor)

    def delete_project(self, actor: dict[str, Any], project_id: int) -> dict[str, Any]:
        """Delete one collaboration project and only its managed image copy."""
        if actor["role"] != "admin":
            raise AnnotationError("只有管理员可以删除项目。", 403)
        project = self.get_project(project_id, actor)
        project_dir = self.projects_dir / str(project_id)
        expected_parent = self.projects_dir.resolve()
        if project_dir.is_symlink():
            raise AnnotationError("项目数据目录异常，已拒绝删除。", 409)
        if project_dir.exists() and project_dir.resolve().parent != expected_parent:
            raise AnnotationError("项目数据目录异常，已拒绝删除。", 409)

        staged_dir: Path | None = None
        if project_dir.exists():
            trash_dir = self.workspace / ".trash"
            trash_dir.mkdir(parents=True, exist_ok=True)
            staged_dir = trash_dir / f"project-{project_id}-{uuid.uuid4().hex}"
            project_dir.replace(staged_dir)
        try:
            with self.connect() as connection:
                cursor = connection.execute("DELETE FROM projects WHERE id=?", (project_id,))
                if cursor.rowcount != 1:
                    raise AnnotationError("项目不存在。", 404)
        except Exception:
            if staged_dir and staged_dir.exists() and not project_dir.exists():
                staged_dir.replace(project_dir)
            raise
        if staged_dir and staged_dir.exists():
            shutil.rmtree(staged_dir)
        return {
            "id": project_id,
            "name": project["name"],
            "platform_project_id": project.get("platform_project_id") or "",
            "deleted_managed_copy": True,
        }

    def import_uploaded_image(self, actor: dict[str, Any], project_id: int, filename: str, raw: bytes) -> dict[str, Any]:
        if actor["role"] not in {"admin", "reviewer"}:
            raise AnnotationError("只有管理员或审核员可以上传图片。", 403)
        self.get_project(project_id, actor)
        normalized = str(filename or "").replace("\\", "/").strip("/")
        relative = PurePosixPath(normalized)
        if not normalized or relative.is_absolute() or ".." in relative.parts:
            raise AnnotationError("图片文件名无效。")
        suffix = Path(relative.name).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            raise AnnotationError("图片格式不受支持。")
        try:
            with Image.open(BytesIO(raw)) as image:
                width, height = image.size
                image.verify()
        except (OSError, ValueError) as exc:
            raise AnnotationError("上传内容不是可读取的图片。") from exc
        destination = self.projects_dir / str(project_id) / "images"
        destination.mkdir(parents=True, exist_ok=True)
        stored_name = f"{uuid.uuid4().hex}{suffix}"
        destination_path = destination / stored_name
        now = _now()
        with self.connect() as connection:
            if connection.execute(
                "SELECT 1 FROM items WHERE project_id=? AND relative_source=?",
                (project_id, relative.as_posix()),
            ).fetchone():
                raise AnnotationError("该项目中已存在同名图片。", 409)
            destination_path.write_bytes(raw)
            cursor = connection.execute(
                """INSERT INTO items(
                       project_id,original_name,relative_source,stored_name,width,height,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?)""",
                (project_id, relative.name, relative.as_posix(), stored_name, width, height, now, now),
            )
            connection.execute("UPDATE projects SET updated_at=? WHERE id=?", (now, project_id))
            self._event(connection, project_id, int(cursor.lastrowid), actor["id"], "image_uploaded", {"source": relative.as_posix()})
        return {"id": int(cursor.lastrowid), "name": relative.name, "relative_source": relative.as_posix()}

    def _project_access_clause(self, actor: dict[str, Any]) -> tuple[str, tuple[Any, ...]]:
        # Every signed-in member can see the project list. Annotators only see
        # their own images plus the unclaimed queue inside each project.
        return "", ()

    def list_projects(self, actor: dict[str, Any]) -> list[dict[str, Any]]:
        clause, params = self._project_access_clause(actor)
        with self.connect() as connection:
            rows = connection.execute(
                f"""SELECT p.*,
                    COUNT(i.id) AS item_count,
                    SUM(CASE WHEN i.status='approved' THEN 1 ELSE 0 END) AS approved_count,
                    SUM(CASE WHEN i.status='submitted' THEN 1 ELSE 0 END) AS submitted_count,
                    SUM(CASE WHEN i.status IN ('assigned','in_progress','rejected') THEN 1 ELSE 0 END) AS active_count
                    FROM projects p LEFT JOIN items i ON i.project_id=p.id
                    {clause} GROUP BY p.id ORDER BY p.updated_at DESC""",
                params,
            ).fetchall()
        projects = [self._project_public(row) for row in rows]
        if actor["role"] == "annotator":
            for project in projects:
                project.pop("source_root", None)
        return projects

    def _project_public(self, row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["labels"] = _parse_json(result.pop("labels_json", "[]"), [])
        result["review_enabled"] = bool(result.get("review_enabled"))
        for key in ("item_count", "approved_count", "submitted_count", "active_count"):
            result[key] = int(result.get(key) or 0)
        return result

    def get_project(self, project_id: int, actor: dict[str, Any]) -> dict[str, Any]:
        projects = {item["id"]: item for item in self.list_projects(actor)}
        if project_id not in projects:
            raise AnnotationError("项目不存在或没有访问权限。", 404)
        return projects[project_id]

    def list_items(self, project_id: int, actor: dict[str, Any], status: str = "") -> list[dict[str, Any]]:
        self.get_project(project_id, actor)
        conditions = ["i.project_id=?"]
        params: list[Any] = [project_id]
        if actor["role"] == "annotator":
            conditions.append("(i.assignee_id=? OR (i.assignee_id IS NULL AND i.status='unassigned'))")
            params.append(actor["id"])
        if status == "todo":
            conditions.append("i.status IN ('unassigned','assigned','in_progress','rejected')")
        elif status and status in STATUSES:
            conditions.append("i.status=?")
            params.append(status)
        with self.connect() as connection:
            rows = connection.execute(
                f"""SELECT i.id,i.original_name,i.relative_source,i.width,i.height,i.assignee_id,
                    i.status,i.revision,i.lock_user_id,i.lock_expires_at,i.review_comment,i.updated_at,
                    u.username AS assignee_name
                    FROM items i LEFT JOIN users u ON u.id=i.assignee_id
                    WHERE {' AND '.join(conditions)}
                    ORDER BY CASE i.status WHEN 'rejected' THEN 0 WHEN 'in_progress' THEN 1 WHEN 'assigned' THEN 2
                             WHEN 'unassigned' THEN 3 WHEN 'submitted' THEN 4 WHEN 'approved' THEN 5 ELSE 6 END, i.id""",
                params,
            ).fetchall()
        now = _now()
        return [{**dict(row), "locked": bool(row["lock_user_id"] and (row["lock_expires_at"] or 0) > now)} for row in rows]

    def acquire_item(self, item_id: int, actor: dict[str, Any]) -> dict[str, Any]:
        now = _now()
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT i.*,p.labels_json,p.name AS project_name,u.username AS assignee_name
                   FROM items i JOIN projects p ON p.id=i.project_id
                   LEFT JOIN users u ON u.id=i.assignee_id WHERE i.id=?""",
                (item_id,),
            ).fetchone()
            if row is None:
                raise AnnotationError("图片任务不存在。", 404)
            if actor["role"] == "annotator" and row["assignee_id"] not in (None, actor["id"]):
                raise AnnotationError("这张图片已经由其他成员领取。", 403)
            if actor["role"] == "annotator" and row["assignee_id"] is None and row["status"] != "unassigned":
                raise AnnotationError("这张图片当前不能领取。", 403)
            if actor["role"] == "annotator" and row["status"] in {"submitted", "approved"}:
                connection.commit()
                return self.item_detail(item_id, actor)
            locked_by_other = row["lock_user_id"] not in (None, actor["id"]) and (row["lock_expires_at"] or 0) > now
            if locked_by_other:
                owner = connection.execute("SELECT username FROM users WHERE id=?", (row["lock_user_id"],)).fetchone()
                raise AnnotationError(f"该图片正在由 {owner['username'] if owner else '其他成员'} 编辑，请稍后再试。", 409)
            connection.execute(
                """UPDATE items SET
                       assignee_id=CASE WHEN assignee_id IS NULL AND ?='annotator' THEN ? ELSE assignee_id END,
                       lock_user_id=?,lock_expires_at=?,
                       status=CASE WHEN status IN ('unassigned','assigned') THEN 'in_progress' ELSE status END,
                       updated_at=? WHERE id=?""",
                (actor["role"], actor["id"], actor["id"], now + LOCK_SECONDS, now, item_id),
            )
            connection.commit()
        finally:
            connection.close()
        return self.item_detail(item_id, actor)

    def item_detail(self, item_id: int, actor: dict[str, Any]) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT i.*,p.labels_json,p.name AS project_name,u.username AS assignee_name
                   FROM items i JOIN projects p ON p.id=i.project_id
                   LEFT JOIN users u ON u.id=i.assignee_id WHERE i.id=?""",
                (item_id,),
            ).fetchone()
        if row is None:
            raise AnnotationError("图片任务不存在。", 404)
        if actor["role"] == "annotator" and row["assignee_id"] != actor["id"]:
            raise AnnotationError("这张图片没有分配给你。", 403)
        result = dict(row)
        result["labels"] = _parse_json(result.pop("labels_json"), [])
        result["boxes"] = _parse_json(result.pop("annotations_json"), [])
        result["image_url"] = f"/api/image?id={item_id}"
        result["locked_by_me"] = result["lock_user_id"] == actor["id"] and (result["lock_expires_at"] or 0) > _now()
        return result

    def image_path(self, item_id: int, actor: dict[str, Any]) -> tuple[Path, str]:
        item = self.item_detail(item_id, actor)
        path = (self.projects_dir / str(item["project_id"]) / "images" / item["stored_name"]).resolve()
        expected = (self.projects_dir / str(item["project_id"]) / "images").resolve()
        if expected not in path.parents or not path.is_file():
            raise AnnotationError("图片文件不存在。", 404)
        return path, item["original_name"]

    def _validate_boxes(self, raw_boxes: Any, item: sqlite3.Row, labels: list[str]) -> list[dict[str, Any]]:
        if not isinstance(raw_boxes, list):
            raise AnnotationError("标注数据格式无效。")
        boxes: list[dict[str, Any]] = []
        for raw in raw_boxes[:5000]:
            if not isinstance(raw, dict):
                continue
            label = str(raw.get("label", "")).strip()
            if label not in labels:
                raise AnnotationError(f"未知类别：{label}")
            try:
                x = max(0.0, min(float(raw["x"]), float(item["width"])))
                y = max(0.0, min(float(raw["y"]), float(item["height"])))
                width = max(0.0, min(float(raw["w"]), float(item["width"]) - x))
                height = max(0.0, min(float(raw["h"]), float(item["height"]) - y))
            except (KeyError, TypeError, ValueError):
                raise AnnotationError("检测框坐标无效。")
            if width < 2 or height < 2:
                continue
            boxes.append({"id": str(raw.get("id") or uuid.uuid4().hex), "label": label, "x": round(x, 2), "y": round(y, 2), "w": round(width, 2), "h": round(height, 2)})
        return boxes

    def save_item(self, item_id: int, actor: dict[str, Any], boxes: Any, revision: int, submit: bool) -> dict[str, Any]:
        now = _now()
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT i.*,p.labels_json,p.review_enabled FROM items i JOIN projects p ON p.id=i.project_id WHERE i.id=?""",
                (item_id,),
            ).fetchone()
            if row is None:
                raise AnnotationError("图片任务不存在。", 404)
            if actor["role"] == "annotator" and row["assignee_id"] != actor["id"]:
                raise AnnotationError("这张图片没有分配给你。", 403)
            if actor["role"] == "annotator" and row["status"] in {"submitted", "approved"}:
                raise AnnotationError("该任务已经提交，需由审核员驳回后才能继续编辑。", 409)
            if row["lock_user_id"] != actor["id"] or (row["lock_expires_at"] or 0) <= now:
                raise AnnotationError("编辑锁已失效，请重新打开图片后再保存。", 409)
            if int(row["revision"]) != int(revision):
                raise AnnotationError("标注已被更新，请重新载入后再保存。", 409)
            clean_boxes = self._validate_boxes(boxes, row, _parse_json(row["labels_json"], []))
            status = ("submitted" if row["review_enabled"] else "approved") if submit else "in_progress"
            lock_user = None if submit else actor["id"]
            lock_expiry = None if submit else now + LOCK_SECONDS
            connection.execute(
                """UPDATE items SET annotations_json=?,revision=revision+1,status=?,lock_user_id=?,
                   lock_expires_at=?,review_comment='',updated_at=? WHERE id=?""",
                (_json(clean_boxes), status, lock_user, lock_expiry, now, item_id),
            )
            action = "submitted" if status == "submitted" else "completed" if status == "approved" else "saved"
            self._event(connection, row["project_id"], item_id, actor["id"], action, {"boxes": len(clean_boxes)})
            connection.commit()
        finally:
            connection.close()
        return self.item_detail(item_id, actor)

    def assign_items(
        self,
        actor: dict[str, Any],
        project_id: int,
        assignee_id: int,
        count: int = 0,
        item_ids: Iterable[int] = (),
    ) -> int:
        if actor["role"] not in {"admin", "reviewer"}:
            raise AnnotationError("没有分配任务的权限。", 403)
        self.get_project(project_id, actor)
        with self.connect() as connection:
            assignee = connection.execute("SELECT id,active FROM users WHERE id=?", (assignee_id,)).fetchone()
            if assignee is None or not assignee["active"]:
                raise AnnotationError("被分配成员不存在或已停用。")
            selected = [int(item) for item in item_ids if str(item).isdigit()]
            if not selected:
                limit = max(1, min(int(count or 1), 10_000))
                selected = [row["id"] for row in connection.execute(
                    "SELECT id FROM items WHERE project_id=? AND status='unassigned' ORDER BY id LIMIT ?",
                    (project_id, limit),
                ).fetchall()]
            if not selected:
                return 0
            placeholders = ",".join("?" for _ in selected)
            now = _now()
            cursor = connection.execute(
                f"""UPDATE items SET assignee_id=?,status='assigned',lock_user_id=NULL,lock_expires_at=NULL,
                    review_comment='',updated_at=? WHERE project_id=? AND id IN ({placeholders})""",
                (assignee_id, now, project_id, *selected),
            )
            assigned = cursor.rowcount
            self._event(connection, project_id, None, actor["id"], "tasks_assigned", {"assignee_id": assignee_id, "count": assigned})
        return assigned

    def review_item(self, item_id: int, actor: dict[str, Any], approve: bool, comment: str = "") -> dict[str, Any]:
        if actor["role"] not in {"admin", "reviewer"}:
            raise AnnotationError("只有管理员或审核员可以审核。", 403)
        comment = str(comment or "").strip()[:1000]
        with self.connect() as connection:
            row = connection.execute("SELECT project_id,status FROM items WHERE id=?", (item_id,)).fetchone()
            if row is None:
                raise AnnotationError("图片任务不存在。", 404)
            if row["status"] != "submitted":
                raise AnnotationError("只有已提交的任务可以审核。")
            status = "approved" if approve else "rejected"
            connection.execute(
                "UPDATE items SET status=?,review_comment=?,lock_user_id=NULL,lock_expires_at=NULL,updated_at=? WHERE id=?",
                (status, comment, _now(), item_id),
            )
            self._event(connection, row["project_id"], item_id, actor["id"], status, {"comment": comment})
        return self.item_detail(item_id, actor)

    def release_item(self, item_id: int, actor: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE items SET lock_user_id=NULL,lock_expires_at=NULL WHERE id=? AND lock_user_id=?",
                (item_id, actor["id"]),
            )

    def export_rows(self, project_id: int, actor: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        project = self.get_project(project_id, actor)
        if actor["role"] not in {"admin", "reviewer"}:
            raise AnnotationError("只有管理员或审核员可以导出数据集。", 403)
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM items WHERE project_id=? AND status='approved' ORDER BY id",
                (project_id,),
            ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["boxes"] = _parse_json(item.pop("annotations_json"), [])
            item["path"] = self.projects_dir / str(project_id) / "images" / item["stored_name"]
            items.append(item)
        if not items:
            raise AnnotationError("还没有审核通过的图片，暂时不能导出。")
        return project, items

    def dataset_version(self, project: dict[str, Any], items: list[dict[str, Any]]) -> str:
        digest = hashlib.sha256(_json(project["labels"]).encode("utf-8"))
        for item in items:
            digest.update(f"{item['id']}|{item['relative_source']}|{item['revision']}\n".encode("utf-8"))
        return digest.hexdigest()[:12]

    def _event(
        self,
        connection: sqlite3.Connection,
        project_id: int | None,
        item_id: int | None,
        user_id: int | None,
        action: str,
        detail: dict[str, Any],
    ) -> None:
        connection.execute(
            "INSERT INTO events(project_id,item_id,user_id,action,detail_json,created_at) VALUES(?,?,?,?,?,?)",
            (project_id, item_id, user_id, action, _json(detail), _now()),
        )
