"""Shared multi-account credential pool storage, ordering, cooldowns, and rotation.

Every file-backed provider credential store (`oauth.json` or `credentials.json`)
shares one on-disk envelope (`{"version": 1, "accounts": [...]}`), atomic `0o600`
locked writes, backward-compatible migration of legacy single-record dictionaries,
`user_scope` -> deployment/org store ordering, and process-wide usage-limit
cooldown tracking (`ProviderCredentialPool`). Provider-specific auth fields
(`project_id`, `refresh_token`, `api_key`, OAuth token endpoints) remain owned by
each provider module.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import secrets
import time
from typing import Any, Callable, Generator, Mapping, Sequence

LOGGER = logging.getLogger(__name__)

CREDENTIAL_POOL_STORE_VERSION = 1
USER_SCOPE_FIELD = "auth_user_scope"
ACCOUNT_IDENTITY_FIELD = "account_identity"
UNDECLARED_LIMIT_BLOCK_SECONDS = 300.0
REFRESH_FAILURE_BLOCK_SECONDS = 300.0


def normalize_user_scope(value: Any) -> str:
  """Normalize a positive `risk_user_id` scope; empty string means the org pool."""
  if isinstance(value, bool) or value is None:
    return ""
  text = str(value).strip()
  if not text:
    return ""
  try:
    scope = int(text)
  except (TypeError, ValueError):
    return text
  return str(scope) if scope > 0 else ""


def store_base_dir(environ: Mapping[str, str] | None = None) -> Path:
  """The root directory for provider credential stores (`$USER_DATA_DIR` or `~/.agent_gateway`)."""
  env = os.environ if environ is None else environ
  user_data = str(env.get("USER_DATA_DIR") or "").strip()
  return Path(user_data).expanduser() if user_data else Path.home() / ".agent_gateway"


def resolve_provider_store_path(
  family: str,
  filename: str,
  config: Mapping[str, Any] | None = None,
  *,
  environ: Mapping[str, str] | None = None,
  user_scope: Any = None,
  override_config_keys: Sequence[str] = ("auth_store_path",),
  override_env_vars: Sequence[str] = (),
) -> Path:
  """Resolve a provider store path for `user_scope` or the deployment org store.

  A named `user_scope` resolves `$USER_DATA_DIR/users/<scope>/<family>/<filename>`
  (or `~/.agent_gateway/users/<scope>/<family>/<filename>` when `USER_DATA_DIR` is
  unset), unless an explicit `override_config_keys` entry was passed by the caller.
  Environment variable overrides (`override_env_vars`) name the deployment org
  store path and never collapse a scoped user store onto the org file.
  """
  cfg = config or {}
  env = os.environ if environ is None else environ
  for key in override_config_keys:
    raw_override = str(cfg.get(key) or "").strip()
    if raw_override:
      return Path(raw_override).expanduser()

  scope = normalize_user_scope(
    user_scope if user_scope is not None else cfg.get(USER_SCOPE_FIELD)
  )
  base = store_base_dir(env)
  if scope:
    return base / "users" / scope / family / filename

  for env_name in override_env_vars:
    env_override = str(env.get(env_name) or "").strip()
    if env_override:
      return Path(env_override).expanduser()
  return base / family / filename


def _infer_base_from_store_path(path: Path, family: str, filename: str) -> Path | None:
  """If `path` is `<base>/users/<scope>/<family>/<filename>` or `<base>/<family>/<filename>`, return `<base>`.

  A path whose filename differs from the canonical one is an explicit store
  file, not a family directory: its siblings live in that file, never under
  the canonical name.
  """
  parts = path.parts
  if not parts or parts[-1] != filename:
    return None
  if len(parts) >= 4 and parts[-4] == "users" and parts[-2] == family:
    return path.parents[3]
  if len(parts) >= 2 and parts[-2] == family:
    return path.parents[1]
  return None


def resolve_provider_store_paths(
  family: str,
  filename: str,
  config: Mapping[str, Any] | None = None,
  *,
  environ: Mapping[str, str] | None = None,
  user_scope: Any = None,
  override_config_keys: Sequence[str] = ("auth_store_path",),
  override_env_vars: Sequence[str] = (),
) -> tuple[Path | None, Path]:
  """Return `(user_store_path, org_store_path)` for `user_scope` -> org ordering."""
  cfg = config or {}
  env = os.environ if environ is None else environ
  scope = normalize_user_scope(
    user_scope if user_scope is not None else cfg.get(USER_SCOPE_FIELD)
  )
  for key in override_config_keys:
    raw_override = str(cfg.get(key) or "").strip()
    if raw_override:
      override_path = Path(raw_override).expanduser()
      inferred_base = _infer_base_from_store_path(override_path, family, filename)
      if inferred_base is not None:
        org_path = inferred_base / family / filename
        for env_name in override_env_vars:
          env_override = str(env.get(env_name) or "").strip()
          if env_override:
            org_path = Path(env_override).expanduser()
            break
        if not scope:
          return None, org_path
        user_path = inferred_base / "users" / scope / family / filename
        return (None, org_path) if user_path == org_path else (user_path, org_path)
      return None, override_path

  org_path = resolve_provider_store_path(
    family,
    filename,
    cfg,
    environ=env,
    user_scope="",
    override_config_keys=(),
    override_env_vars=override_env_vars,
  )
  if not scope:
    return None, org_path
  user_path = resolve_provider_store_path(
    family,
    filename,
    cfg,
    environ=env,
    user_scope=scope,
    override_config_keys=(),
    override_env_vars=override_env_vars,
  )
  if user_path == org_path:
    return None, org_path
  return user_path, org_path

@contextlib.contextmanager
def store_write_lock(path: Path) -> Generator[None, None, None]:
  """Serialize store mutations across processes using a `0o600` sibling lockfile."""
  path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
  try:
    path.parent.chmod(0o700)
  except OSError:
    pass
  lock_path = path.with_name(f".{path.name}.lock")
  fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
  try:
    try:
      os.fchmod(fd, 0o600)
    except OSError:
      pass
    fcntl.flock(fd, fcntl.LOCK_EX)
    yield
  finally:
    try:
      fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
      os.close(fd)


@contextlib.contextmanager
def store_read_lock(path: Path) -> Generator[None, None, None]:
  """Acquire a non-blocking shared lock when available; otherwise read the atomic file snapshot.

  All store writes use atomic temp-file replacement (`os.replace`), so `path`
  is always a complete JSON snapshot on POSIX. Using `LOCK_SH | LOCK_NB`
  prevents a synchronous pool read from blocking the event loop while an async
  token refresh coroutine on the same loop holds `LOCK_EX` across an HTTP await.
  """
  lock_path = path.with_name(f".{path.name}.lock")
  try:
    fd = os.open(lock_path, os.O_RDONLY)
  except OSError:
    yield
    return
  acquired = False
  try:
    try:
      fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
      acquired = True
    except BlockingIOError:
      pass
    yield
  finally:
    try:
      if acquired:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
      os.close(fd)

def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
  """Write `payload` atomically to `path` with `0o600` permissions and `fsync`."""
  path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
  try:
    path.parent.chmod(0o700)
  except OSError:
    pass
  temp_path = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
  fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
      json.dump(dict(payload), handle, indent=2, sort_keys=True)
      handle.write("\n")
      handle.flush()
      os.fsync(handle.fileno())
    os.chmod(temp_path, 0o600)
    os.replace(temp_path, path)
    os.chmod(path, 0o600)
    try:
      dir_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
      dir_fd = None
    if dir_fd is not None:
      try:
        os.fsync(dir_fd)
      except OSError:
        pass
      finally:
        os.close(dir_fd)
  finally:
    try:
      temp_path.unlink()
    except FileNotFoundError:
      pass


def derive_account_identity(
  record: Mapping[str, Any],
  *,
  prefix: str = "account",
  material_keys: Sequence[str] = ("refresh_token", "access_token", "api_key"),
) -> str:
  """Return a stable account identity for `record`, deriving a digest label if unnamed."""
  for key in ("identity", "email", "account_id", "user_email", "user_id", "project_id"):
    val = str(record.get(key) or "").strip()
    if val:
      return val
  for key in material_keys:
    raw = str(record.get(key) or "").strip()
    if raw:
      digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
      return f"{prefix}-{digest}"
  return f"{prefix}-default"


def _normalize_account_entry(
  entry: Mapping[str, Any],
  *,
  default_prefix: str,
  required_any_keys: Sequence[str],
) -> dict[str, Any] | None:
  if not any(str(entry.get(k) or "").strip() for k in required_any_keys):
    return None
  normalized = dict(entry)
  normalized["identity"] = derive_account_identity(
    normalized,
    prefix=default_prefix,
    material_keys=required_any_keys,
  )
  return normalized


def _load_pool_accounts_unlocked(
  path: Path,
  *,
  default_prefix: str = "account",
  required_any_keys: Sequence[str] = ("access_token", "refresh_token", "api_key"),
  strict: bool = False,
) -> list[dict[str, Any]]:
  try:
    raw = path.read_text(encoding="utf-8")
  except FileNotFoundError:
    return []
  except OSError as exc:
    if strict:
      raise RuntimeError(f"Unable to read credential store: {path}") from exc
    return []
  try:
    parsed = json.loads(raw)
  except json.JSONDecodeError as exc:
    if strict:
      raise RuntimeError(f"Invalid credential store JSON: {path}") from exc
    return []
  if not isinstance(parsed, dict):
    if strict:
      raise RuntimeError(f"Invalid credential store payload: {path}")
    return []

  accounts_raw = parsed.get("accounts")
  records: list[dict[str, Any]] = []
  if isinstance(accounts_raw, list):
    for item in accounts_raw:
      if isinstance(item, Mapping):
        norm = _normalize_account_entry(
          item,
          default_prefix=default_prefix,
          required_any_keys=required_any_keys,
        )
        if norm is not None:
          records.append(norm)
    return records

  single = _normalize_account_entry(
    parsed,
    default_prefix=default_prefix,
    required_any_keys=required_any_keys,
  )
  if single is not None:
    records.append(single)
  return records


def load_pool_accounts(
  path: Path,
  *,
  default_prefix: str = "account",
  required_any_keys: Sequence[str] = ("access_token", "refresh_token", "api_key"),
  strict: bool = False,
) -> list[dict[str, Any]]:
  """Load all valid accounts from `path`, transparently reading legacy single-record dicts."""
  with store_read_lock(path):
    return _load_pool_accounts_unlocked(
      path,
      default_prefix=default_prefix,
      required_any_keys=required_any_keys,
      strict=strict,
    )


def write_pool_accounts_unlocked(
  path: Path,
  accounts: Sequence[Mapping[str, Any]],
  *,
  extra_top_level: Mapping[str, Any] | None = None,
) -> None:
  """Write the multi-account store envelope while preserving primary single-record fields."""
  normalized_list = [dict(a) for a in accounts]
  payload: dict[str, Any] = {}
  if extra_top_level:
    payload.update(extra_top_level)
  elif normalized_list:
    for key, val in normalized_list[0].items():
      payload.setdefault(key, val)
  payload["version"] = CREDENTIAL_POOL_STORE_VERSION
  payload["accounts"] = normalized_list
  atomic_write_json(path, payload)


def upsert_pool_account_unlocked(
  path: Path,
  record: Mapping[str, Any],
  *,
  default_prefix: str = "account",
  required_any_keys: Sequence[str] = ("access_token", "refresh_token", "api_key"),
) -> list[dict[str, Any]]:
  """`upsert_pool_account` for a caller already holding `store_write_lock(path)`.

  `store_write_lock` is a per-open-file flock, so re-acquiring it from the same
  process on a new fd blocks against the held one.
  """
  entry = _normalize_account_entry(
    record,
    default_prefix=default_prefix,
    required_any_keys=required_any_keys,
  )
  if entry is None:
    raise ValueError(
      f"Credential record for {path} is missing required material ({', '.join(required_any_keys)})"
    )
  target_id = entry["identity"]
  accounts = _load_pool_accounts_unlocked(
    path,
    default_prefix=default_prefix,
    required_any_keys=required_any_keys,
    strict=False,
  )
  replaced = False
  for idx, existing in enumerate(accounts):
    if existing.get("identity") == target_id:
      merged = dict(existing)
      merged.update(entry)
      accounts[idx] = merged
      replaced = True
      break
  if not replaced:
    accounts.append(entry)
  write_pool_accounts_unlocked(path, accounts)
  return accounts


def upsert_pool_account(
  path: Path,
  record: Mapping[str, Any],
  *,
  default_prefix: str = "account",
  required_any_keys: Sequence[str] = ("access_token", "refresh_token", "api_key"),
) -> list[dict[str, Any]]:
  """Insert or update `record` by its `identity` in `path` without overwriting other seats."""
  with store_write_lock(path):
    return upsert_pool_account_unlocked(
      path,
      record,
      default_prefix=default_prefix,
      required_any_keys=required_any_keys,
    )


def remove_pool_account(
  path: Path,
  identity: str | None = None,
  *,
  default_prefix: str = "account",
  required_any_keys: Sequence[str] = ("access_token", "refresh_token", "api_key"),
) -> bool:
  """Remove a single account by `identity`, or unlink the entire store when `identity` is empty."""
  norm_id = str(identity or "").strip()
  if not norm_id:
    # Whole-store removal is a mutation and must serialize with writers.
    with store_write_lock(path):
      try:
        path.unlink()
        return True
      except FileNotFoundError:
        return False
  with store_write_lock(path):
    accounts = _load_pool_accounts_unlocked(
      path,
      default_prefix=default_prefix,
      required_any_keys=required_any_keys,
      strict=False,
    )
    kept = [a for a in accounts if a.get("identity") != norm_id]
    if len(kept) == len(accounts):
      return False
    if not kept:
      try:
        path.unlink()
      except FileNotFoundError:
        pass
      return True
    write_pool_accounts_unlocked(path, kept)
    return True


@dataclass(frozen=True)
class PoolAccountCandidate:
  """One resolved candidate in a provider's ordered credential pool."""

  provider: str
  key: str = field(repr=False)
  record: dict[str, Any] = field(repr=False)
  store_path: Path | None = None
  is_user_store: bool = False
  auth_mode: str = "oauth"

  @property
  def account_identity(self) -> str:
    return str(self.record.get("identity") or self.key)

  @property
  def token(self) -> str:
    if self.auth_mode == "api":
      return str(self.record.get("api_key") or self.record.get("access_token") or "").strip()
    return str(self.record.get("access_token") or self.record.get("api_key") or "").strip()


class ProviderCredentialPool:
  """Process-wide usage-limit and refresh-failure cooldown tracker for one provider."""

  def __init__(self) -> None:
    self._blocked_until: dict[str, float] = {}

  def blocked_until(self, key: str) -> float:
    return self._blocked_until.get(str(key or "").strip(), 0.0)

  def block(self, key: str, *, until: float) -> None:
    normalized = str(key or "").strip()
    if not normalized:
      return
    self._blocked_until[normalized] = max(
      self._blocked_until.get(normalized, 0.0),
      float(until),
    )

  def first_unblocked(self, keys: Sequence[str], *, now: float | None = None) -> str:
    moment = time.time() if now is None else now
    for key in keys:
      if self.blocked_until(key) <= moment:
        return str(key)
    return ""

  def clear(self) -> None:
    self._blocked_until.clear()


_PROVIDER_POOLS: dict[str, ProviderCredentialPool] = {}


def get_provider_credential_pool(provider: str) -> ProviderCredentialPool:
  """Return the process-wide `ProviderCredentialPool` for `provider`."""
  norm = str(provider or "").strip().lower()
  pool = _PROVIDER_POOLS.get(norm)
  if pool is None:
    pool = ProviderCredentialPool()
    _PROVIDER_POOLS[norm] = pool
  return pool


def select_pool_candidate(
  candidates: Sequence[PoolAccountCandidate],
  *,
  pool: ProviderCredentialPool,
  refresh_fn: Callable[[PoolAccountCandidate], PoolAccountCandidate] | None = None,
  now: float | None = None,
) -> PoolAccountCandidate | None:
  """Select the first unblocked candidate (refreshing if needed), or fall back to candidate[0]."""
  if not candidates:
    return None
  moment = time.time() if now is None else now
  for candidate in candidates:
    if pool.blocked_until(candidate.key) > moment:
      continue
    if refresh_fn is not None:
      try:
        return refresh_fn(candidate)
      except Exception as exc:
        pool.block(candidate.key, until=moment + REFRESH_FAILURE_BLOCK_SECONDS)
        LOGGER.warning(
          "%s credential refresh failed for %s; parking for %.0fs: %s",
          candidate.provider,
          candidate.account_identity,
          REFRESH_FAILURE_BLOCK_SECONDS,
          exc,
        )
        continue
    return candidate
  return candidates[0]


def rotate_pool_candidate(
  candidates: Sequence[PoolAccountCandidate],
  *,
  current_key: str,
  until: float,
  pool: ProviderCredentialPool,
  refresh_fn: Callable[[PoolAccountCandidate], PoolAccountCandidate] | None = None,
  now: float | None = None,
) -> PoolAccountCandidate | None:
  """Park `current_key` until `until` and return the next unblocked sibling, or `None`."""
  moment = time.time() if now is None else now
  norm_current = str(current_key or "").strip()
  matched_key = norm_current
  for candidate in candidates:
    if candidate.key == norm_current or candidate.token == norm_current or candidate.account_identity == norm_current:
      matched_key = candidate.key
      break
  if matched_key:
    pool.block(matched_key, until=until)
  for candidate in candidates:
    if candidate.key == matched_key:
      continue
    if pool.blocked_until(candidate.key) > moment:
      continue
    if refresh_fn is not None:
      try:
        return refresh_fn(candidate)
      except Exception as exc:
        pool.block(candidate.key, until=moment + REFRESH_FAILURE_BLOCK_SECONDS)
        LOGGER.warning(
          "%s credential refresh failed during rotation for %s; parking for %.0fs: %s",
          candidate.provider,
          candidate.account_identity,
          REFRESH_FAILURE_BLOCK_SECONDS,
          exc,
        )
        continue
    return candidate
  return None


PoolCandidateResolver = Callable[
  [Mapping[str, Any]],
  tuple[
    list[PoolAccountCandidate],
    Callable[[PoolAccountCandidate], PoolAccountCandidate] | None,
  ],
]
PoolCandidateMaterializer = Callable[[PoolAccountCandidate, Mapping[str, Any]], dict[str, Any]]

_PROVIDER_POOL_HOOKS: dict[str, tuple[PoolCandidateResolver, PoolCandidateMaterializer]] = {}


def register_provider_pool_hooks(
  provider: str,
  *,
  resolver: PoolCandidateResolver,
  materializer: PoolCandidateMaterializer,
) -> None:
  """Register `provider`'s candidate resolver and materializer for mid-run rotation."""
  _PROVIDER_POOL_HOOKS[str(provider or "").strip().lower()] = (resolver, materializer)


def get_provider_pool_hooks(
  provider: str,
) -> tuple[PoolCandidateResolver, PoolCandidateMaterializer] | None:
  return _PROVIDER_POOL_HOOKS.get(str(provider or "").strip().lower())


def rotate_provider_credential(
  provider: str,
  config: Mapping[str, Any],
  *,
  failure_kind: str,
  reset_at: float | None = None,
  status_code: int | None = None,
  rotatable_kinds: frozenset[str] = frozenset({"rate_limit", "auth", "quota"}),
  now: float | None = None,
) -> dict[str, Any] | None:
  """Rotate `provider` to the next unblocked sibling credential in its pool."""
  norm_provider = str(provider or "").strip().lower()
  mode = str(config.get("auth_mode") or ("oauth" if config.get("auth_token") else "api")).strip().lower()
  allowed = rotatable_kinds | ({"billing"} if mode == "api" else frozenset())
  if failure_kind not in allowed:
    return None
  hooks = get_provider_pool_hooks(norm_provider)
  if hooks is None:
    return None
  resolver, materializer = hooks
  candidates, refresh_fn = resolver(config)
  if not candidates:
    return None
  current_key = str(
    config.get(ACCOUNT_IDENTITY_FIELD)
    or config.get("auth_token")
    or config.get("api_key")
    or ""
  ).strip()
  if not current_key:
    return None
  moment = time.time() if now is None else now
  blocked_until = (
    reset_at if reset_at is not None else moment + UNDECLARED_LIMIT_BLOCK_SECONDS
  )
  pool = get_provider_credential_pool(norm_provider)
  selected = rotate_pool_candidate(
    candidates,
    current_key=current_key,
    until=blocked_until,
    pool=pool,
    refresh_fn=refresh_fn,
    now=moment,
  )
  if selected is None:
    LOGGER.warning(
      "%s %s failure (status=%s) with no unblocked sibling credential in a pool of %d",
      norm_provider,
      failure_kind,
      status_code,
      len(candidates),
    )
    return None
  return materializer(selected, config)


__all__ = [
  "ACCOUNT_IDENTITY_FIELD",
  "CREDENTIAL_POOL_STORE_VERSION",
  "PoolAccountCandidate",
  "PoolCandidateMaterializer",
  "PoolCandidateResolver",
  "ProviderCredentialPool",
  "REFRESH_FAILURE_BLOCK_SECONDS",
  "UNDECLARED_LIMIT_BLOCK_SECONDS",
  "USER_SCOPE_FIELD",
  "atomic_write_json",
  "derive_account_identity",
  "get_provider_credential_pool",
  "get_provider_pool_hooks",
  "load_pool_accounts",
  "normalize_user_scope",
  "register_provider_pool_hooks",
  "remove_pool_account",
  "resolve_provider_store_path",
  "resolve_provider_store_paths",
  "rotate_pool_candidate",
  "rotate_provider_credential",
  "select_pool_candidate",
  "store_base_dir",
  "store_read_lock",
  "store_write_lock",
  "upsert_pool_account",
  "upsert_pool_account_unlocked",
  "write_pool_accounts_unlocked",
]
