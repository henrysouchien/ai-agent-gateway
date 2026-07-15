from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import time
from typing import Any, Mapping


CLAUDE_SETUP_TOKEN_LIFETIME_SECONDS = 365 * 24 * 60 * 60


def resolve_anthropic_auth_store_path(
  config: Mapping[str, Any] | None = None,
  *,
  environ: Mapping[str, str] | None = None,
) -> Path:
  cfg = config or {}
  env = os.environ if environ is None else environ
  raw_store = str(cfg.get("auth_store_path") or env.get("ANTHROPIC_AUTH_STORE_PATH") or "").strip()
  if raw_store:
    return Path(raw_store).expanduser()
  user_data = str(env.get("USER_DATA_DIR") or "").strip()
  base = Path(user_data).expanduser() if user_data else Path.home() / ".agent_gateway"
  return base / "anthropic" / "oauth.json"


def load_anthropic_oauth_record(path: Path) -> dict[str, Any] | None:
  try:
    raw = path.read_text(encoding="utf-8")
  except FileNotFoundError:
    return None
  except OSError as exc:
    raise RuntimeError(f"Unable to read Anthropic OAuth token store: {path}") from exc
  try:
    parsed = json.loads(raw)
  except json.JSONDecodeError as exc:
    raise RuntimeError(f"Invalid Anthropic OAuth token store JSON: {path}") from exc
  if not isinstance(parsed, dict):
    raise RuntimeError(f"Invalid Anthropic OAuth token store payload: {path}")
  return dict(parsed)


def save_anthropic_oauth_record(path: Path, record: Mapping[str, Any]) -> None:
  token = str(record.get("auth_token") or "").strip()
  if not token:
    raise ValueError("Anthropic OAuth token record is missing auth_token")
  path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
  try:
    path.parent.chmod(0o700)
  except OSError:
    pass
  temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
  fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
      json.dump(dict(record), handle, indent=2, sort_keys=True)
      handle.write("\n")
      handle.flush()
      os.fsync(handle.fileno())
    os.chmod(temp_path, 0o600)
    os.replace(temp_path, path)
    os.chmod(path, 0o600)
  finally:
    try:
      temp_path.unlink()
    except FileNotFoundError:
      pass


def import_claude_setup_token(
  token: str,
  *,
  path: Path,
  now: float | None = None,
) -> dict[str, Any]:
  normalized = str(token or "").strip()
  if not normalized.startswith("sk-ant-oat"):
    raise ValueError("Claude setup token must start with 'sk-ant-oat'")
  created_at = time.time() if now is None else now
  record = {
    "auth_token": normalized,
    "created_at": created_at,
    "expires_at": created_at + CLAUDE_SETUP_TOKEN_LIFETIME_SECONDS,
    "source": "claude-setup-token",
  }
  save_anthropic_oauth_record(path, record)
  return record


def resolve_anthropic_oauth_token(
  config: Mapping[str, Any] | None = None,
  *,
  environ: Mapping[str, str] | None = None,
) -> tuple[str, Path, dict[str, Any] | None]:
  cfg = config or {}
  env = os.environ if environ is None else environ
  path = resolve_anthropic_auth_store_path(cfg, environ=env)
  env_token = str(
    cfg.get("auth_token")
    or env.get("ANTHROPIC_AUTH_TOKEN")
    or env.get("CLAUDE_CODE_OAUTH_TOKEN")
    or ""
  ).strip()
  if env_token:
    return env_token, path, None
  record = load_anthropic_oauth_record(path)
  return str((record or {}).get("auth_token") or "").strip(), path, record


def anthropic_token_store_is_private(path: Path) -> bool:
  try:
    return stat.S_IMODE(path.stat().st_mode) == 0o600
  except OSError:
    return False


def anthropic_token_is_expiring(record: Mapping[str, Any], *, now: float | None = None) -> bool:
  try:
    expires_at = float(record.get("expires_at") or 0)
  except (TypeError, ValueError):
    return False
  return bool(expires_at and expires_at <= (time.time() if now is None else now) + 5 * 24 * 60 * 60)


__all__ = [
  "CLAUDE_SETUP_TOKEN_LIFETIME_SECONDS",
  "anthropic_token_is_expiring",
  "anthropic_token_store_is_private",
  "import_claude_setup_token",
  "load_anthropic_oauth_record",
  "resolve_anthropic_auth_store_path",
  "resolve_anthropic_oauth_token",
  "save_anthropic_oauth_record",
]
