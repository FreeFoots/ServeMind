from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding shell variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip('"').strip("'")


_load_env_file(PROJECT_ROOT / "backend" / ".env")


def _apply_legacy_env_aliases() -> None:
    """Accept existing local installations; explicit SERVEMIND values win."""
    legacy_prefix = "VERICARTDESK_"
    for key, value in tuple(os.environ.items()):
        if key.startswith(legacy_prefix):
            os.environ.setdefault("SERVEMIND_" + key[len(legacy_prefix):], value)


_apply_legacy_env_aliases()


def _env_path(name: str, default: Path) -> Path:
    """Resolve a project-relative path while accepting absolute overrides."""
    value = os.getenv(name)
    if not value:
        return default
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


DATA_ROOT = _env_path(
    "SERVEMIND_DATA_ROOT",
    PROJECT_ROOT / "data" / "MSOM_Data_Driven_Challenge_2020",
)
ORDER_DATA_PATH = _env_path("SERVEMIND_ORDER_DATA_PATH", DATA_ROOT / "JD_order_data.csv")
DELIVERY_DATA_PATH = _env_path("SERVEMIND_DELIVERY_DATA_PATH", DATA_ROOT / "JD_delivery_data.csv")
KNOWLEDGE_ROOT = _env_path("SERVEMIND_KNOWLEDGE_ROOT", PROJECT_ROOT / "knowledge")
SKILLS_ROOT = _env_path("SERVEMIND_SKILLS_ROOT", PROJECT_ROOT / "backend" / "skills")
COMMERCE_DB_PATH = _env_path("SERVEMIND_COMMERCE_DB_PATH", PROJECT_ROOT / "backend" / "runtime" / "commerce.sqlite3")
DATA_VERSION = os.getenv("SERVEMIND_DATA_VERSION", "msom-2018-03")
APP_ENV = os.getenv("SERVEMIND_ENV", "development")
DEMO_ACCOUNT_SWITCH = os.getenv("SERVEMIND_DEMO_ACCOUNT_SWITCH", "false").lower() in {"1", "true", "yes", "on"}
LOG_LEVEL = os.getenv("SERVEMIND_LOG_LEVEL", "INFO").upper()
HOST = os.getenv("SERVEMIND_HOST", "127.0.0.1")
PORT = int(os.getenv("SERVEMIND_PORT", "8317"))
FRONTEND_PORT = int(os.getenv("SERVEMIND_FRONTEND_PORT", "5317"))
POLICY_ID = os.getenv("SERVEMIND_POLICY_ID", "jd_simulation_default")
POLICY_VERSION = os.getenv("SERVEMIND_POLICY_VERSION", "2026-09-23")
ENABLE_TRACING = os.getenv("SERVEMIND_ENABLE_TRACING", "true").lower() in {"1", "true", "yes", "on"}


def configured_cors_origins() -> list[str]:
    """Return configured browser origins, with the local desktop defaults."""
    raw = os.getenv("SERVEMIND_CORS_ORIGINS", "")
    if raw.strip():
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        f"http://localhost:{FRONTEND_PORT}",
        f"http://127.0.0.1:{FRONTEND_PORT}",
        "http://localhost:5318",
        "http://127.0.0.1:5318",
        "tauri://localhost",
        "http://tauri.localhost",
        "https://tauri.localhost",
    ]
