from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
import secrets
import stat
import time
import urllib.parse
from typing import Any, Callable, Iterator, Mapping, Sequence

from .credential_pool import ProviderCredentialPool, _PROVIDER_POOLS


LOGGER = logging.getLogger(__name__)

# This product's own Anthropic OAuth client. Every credential in the pool is a
# grant THIS client minted through its own login; the gateway never reads or
# refreshes a grant another client (Claude Code, a fleet profile, a keychain
# item) owns, because refresh tokens rotate on use and refreshing someone
# else's grant destroys it.
ANTHROPIC_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
ANTHROPIC_OAUTH_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
ANTHROPIC_OAUTH_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
# Anthropic's out-of-band console callback: the browser lands on a page that
# prints `code#state`, which the operator pastes back into the login command.
# There is no local listener, so enrollment works over SSH as well.
ANTHROPIC_OAUTH_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
# Exactly the two scopes this product uses: identify the enrolled account and
# make inference requests. It does not ask for API-key creation.
ANTHROPIC_OAUTH_SCOPE = "user:profile user:inference"
ANTHROPIC_OAUTH_PROFILE_URL = "https://api.anthropic.com/api/oauth/profile"
# The beta the provider already sends on every OAuth request (see
# anthropic_helpers._OAUTH_BETA_SLUGS); the profile endpoint wants it too.
ANTHROPIC_OAUTH_BETA = "oauth-2025-04-20"
ANTHROPIC_OAUTH_SOURCE = "anthropic-oauth-login"
ANTHROPIC_OAUTH_STORE_VERSION = 3
# A source generation's opaque reference; a frozen bind's `credential_ref`
# carries it, so the format is owned here and nowhere else.
_GENERATION_REF_PREFIX = "anthropic-generation-"

# Sibling setup tokens, one per Claude subscription account, as a JSON array of
# strings. Same encoding as the other multi-valued credential env in this
# product (`GATEWAY_USER_KEYS`, .env.example "JSON-valued env var").
ANTHROPIC_AUTH_TOKENS_ENV = "ANTHROPIC_AUTH_TOKENS"
# The single-token forms and the store location this resolver also reads. They
# are named here because this module owns what "the pool" is made of: any
# process that must resolve the same pool (the gateway, an autonomous child it
# spawns) is configured with exactly these names.
ANTHROPIC_AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
CLAUDE_CODE_OAUTH_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"
ANTHROPIC_AUTH_STORE_PATH_ENV = "ANTHROPIC_AUTH_STORE_PATH"
ANTHROPIC_STORE_BASE_ENV = "USER_DATA_DIR"
# The credential material's own field naming the user whose pool it came from
# (a `risk_user_id`). The scope is a property of the material, exactly like
# `provider`: every consumer that re-resolves the pool from an already-bound
# credential — this provider's rotation, the refresh on the binding path, an
# autonomous child reading the stdin credential handoff — resolves the same
# user's pool without having to be told again.
ANTHROPIC_USER_SCOPE_FIELD = "auth_user_scope"
# Only the gateway materializer enrolls the deployment key. This explicit
# generation reference retains membership across rotation without giving
# ambient operator credentials to other pool consumers.
ANTHROPIC_DEPLOYMENT_API_KEY_FIELD = "deployment_api_key_source"
# A limiter that rejects a credential without naming a reset window keeps it out
# of rotation for this long. The window is a representation of "unknown reset",
# not a policy: the next attempt on that credential re-blocks it if it is still
# limited, and a declared 5h/7d reset always wins over this default.
UNDECLARED_LIMIT_BLOCK_SECONDS = 300.0
# An access token this close to expiry is refreshed before it is used.
REFRESH_MARGIN_SECONDS = 120.0
# A record whose refresh failed is parked exactly like a usage-limited one: the
# next sibling serves, and the next attempt on this record retries the refresh.
REFRESH_FAILURE_BLOCK_SECONDS = 300.0
# Refresh runs on the synchronous credential-binding path (`create_client`,
# `get_anthropic_config`), so a hung token endpoint holds the event loop for as
# long as this allows. Bound it to seconds: a refresh that does not answer in
# that window is a failed refresh, which parks the account and lets the next
# sibling serve, and the next attempt retries it.
HTTP_CONNECT_TIMEOUT_SECONDS = 5.0
HTTP_TIMEOUT_SECONDS = 10.0

JSONMapping = Mapping[str, Any]
TokenPoster = Callable[[str, JSONMapping], JSONMapping]
ProfileGetter = Callable[[str, str], JSONMapping]


class AnthropicOAuthError(RuntimeError):
  """A login, refresh, or store operation this product owns did not complete."""


class AnthropicSourceError(AnthropicOAuthError):
  """The exact credential generation is not currently authorized at its owner."""

  code = "credential_unavailable"
  next_call = "python3 -m agent_gateway.cli auth login anthropic"

  def __init__(self, reason: str) -> None:
    self.reason = reason
    super().__init__(f"Anthropic source generation is {reason}; run `{self.next_call}`")


class AnthropicStoreError(AnthropicOAuthError):
  """The credential store could not be read or written where it was expected.

  A read path (`load`, an exact `credential_source` resolve, `ensure_fresh`)
  never needs to create the store, its directory, or its lock file; a store
  it cannot open is this typed error with its own next call, never a raw
  `OSError`. A writer that cannot open the store raises the same typed error.
  """

  code = "store_unreadable"
  next_call = (
    "ensure the credential store path exists and its directory is readable "
    "(writable for enrollment), then retry"
  )

  def __init__(self, path: Path, exc: OSError | None = None) -> None:
    self.reason = "store_unreadable"
    self.path = path
    detail = f": {exc}" if exc is not None else ""
    super().__init__(f"Unable to access Anthropic credential store {path}{detail}")

def _anthropic_store_base(env: Mapping[str, str]) -> Path:
  """The root every Anthropic token store of this deployment lives under."""

  user_data = str(env.get(ANTHROPIC_STORE_BASE_ENV) or "").strip()
  return Path(user_data).expanduser() if user_data else Path.home() / ".agent_gateway"


def anthropic_user_scope(value: Any) -> str:
  """The pool-owning user as a path component: a positive `risk_user_id`.

  The scope is an identity the request already carries and is never read out
  of a token. A value that names no user is the empty scope, whose pool is the
  org credentials — the tail of every user's pool.
  """

  if isinstance(value, bool):
    return ""
  try:
    scope = int(str(value).strip())
  except (TypeError, ValueError):
    return ""
  return str(scope) if scope > 0 else ""


def resolve_anthropic_auth_store_path(
  config: Mapping[str, Any] | None = None,
  *,
  environ: Mapping[str, str] | None = None,
  user_scope: Any = None,
) -> Path:
  """Where enrolled accounts live: one store per user, plus the org store.

  A named user scope resolves that user's own store under the per-user private
  data root (`$USER_DATA_DIR/users/<risk_user_id>/anthropic/oauth.json`, where
  docs/reference/identity-security.md puts every per-user private file). The
  empty scope resolves the deployment's org store, which every user's pool
  ends with. `user_scope` is read from the credential material when the caller
  does not name one, so a bound credential resolves its own pool.
  """

  cfg = config or {}
  env = os.environ if environ is None else environ
  scope = anthropic_user_scope(
    user_scope if user_scope is not None else cfg.get(ANTHROPIC_USER_SCOPE_FIELD)
  )
  if scope:
    return _anthropic_store_base(env) / "users" / scope / "anthropic" / "oauth.json"
  raw_store = str(
    cfg.get("auth_store_path") or env.get(ANTHROPIC_AUTH_STORE_PATH_ENV) or ""
  ).strip()
  if raw_store:
    return Path(raw_store).expanduser()
  return _anthropic_store_base(env) / "anthropic" / "oauth.json"


@dataclass(frozen=True)
class AnthropicOAuthRecord:
  """Current material for one immutable source grant generation.

  `identity` remains the account-level parking key across OAuth refreshes.
  Generation metadata is assigned by the store, not by a reader or consumer.
  """

  identity: str
  access_token: str = field(repr=False)
  refresh_token: str = field(default="", repr=False)
  expires_at: float = 0.0
  created_at: float = 0.0
  updated_at: float = 0.0
  scope: str = ANTHROPIC_OAUTH_SCOPE
  source: str = ANTHROPIC_OAUTH_SOURCE
  last_error: str = field(default="", repr=False)
  last_error_at: float = 0.0
  grant_hash: str = ""
  generation_id: str = ""
  credential_kind: str = "auth_token"

  @property
  def generation_ref(self) -> str:
    return f"{_GENERATION_REF_PREFIX}{self.generation_id}" if self.generation_id else ""

  @property
  def account_identity(self) -> str:
    return self.identity

  def source_metadata(self) -> dict[str, str]:
    return {
      "account_identity": self.identity,
      "grant_hash": self.grant_hash,
      "generation_id": self.generation_id,
      "generation_ref": self.generation_ref,
    }

  def needs_refresh(self, *, now: float | None = None) -> bool:
    if not self.refresh_token:
      return False
    moment = time.time() if now is None else now
    return self.expires_at <= moment + REFRESH_MARGIN_SECONDS

  def to_payload(self) -> dict[str, Any]:
    return {
      "identity": self.identity,
      "access_token": self.access_token,
      "refresh_token": self.refresh_token,
      "expires_at": self.expires_at,
      "created_at": self.created_at,
      "updated_at": self.updated_at,
      "scope": self.scope,
      "source": self.source,
      "last_error": self.last_error,
      "last_error_at": self.last_error_at,
      "grant_hash": self.grant_hash,
      "generation_id": self.generation_id,
      "credential_kind": self.credential_kind,
    }

  @classmethod
  def from_payload(cls, payload: Mapping[str, Any]) -> "AnthropicOAuthRecord | None":
    """A stored account, or None when this entry carries no usable credential.

    Content this deployment wrote is reported and skipped rather than refused:
    one damaged entry must not take the other enrolled accounts down with it.
    """

    identity = str(payload.get("identity") or "").strip()
    access_token = str(payload.get("access_token") or "").strip()
    if not identity or not access_token:
      LOGGER.warning("Anthropic OAuth store entry without identity/access_token; skipping it")
      return None
    return cls(
      identity=identity,
      access_token=access_token,
      refresh_token=str(payload.get("refresh_token") or "").strip(),
      expires_at=_float_or_zero(payload.get("expires_at")),
      created_at=_float_or_zero(payload.get("created_at")),
      updated_at=_float_or_zero(payload.get("updated_at")),
      scope=str(payload.get("scope") or ANTHROPIC_OAUTH_SCOPE).strip(),
      source=str(payload.get("source") or ANTHROPIC_OAUTH_SOURCE).strip(),
      last_error=str(payload.get("last_error") or "").strip(),
      last_error_at=_float_or_zero(payload.get("last_error_at")),
      grant_hash=str(payload.get("grant_hash") or "").strip(),
      generation_id=str(payload.get("generation_id") or "").strip(),
      credential_kind=str(payload.get("credential_kind") or "auth_token").strip(),
    )


@dataclass(frozen=True)
class AnthropicCredential:
  """A pool member: what to send, what to park, and who can refresh it.

  `key` preserves the existing account/environment parking policy. `record`
  carries the same generation that supplied `token`, including after refresh.
  `store_path` locates its owner and is never encoded into a source reference.
  """

  key: str = field(repr=False)
  token: str = field(repr=False)
  record: AnthropicOAuthRecord | None = None
  store_path: Path | None = field(default=None, repr=False)

  @property
  def account_identity(self) -> str:
    return self.record.account_identity if self.record is not None else ""

  @property
  def grant_hash(self) -> str:
    return self.record.grant_hash if self.record is not None else ""

  @property
  def generation_id(self) -> str:
    return self.record.generation_id if self.record is not None else ""

  @property
  def generation_ref(self) -> str:
    return self.record.generation_ref if self.record is not None else ""

  def source_metadata(self) -> dict[str, str]:
    if self.record is None:
      raise AnthropicSourceError("unregistered")
    return self.record.source_metadata()


@dataclass(frozen=True)
class AnthropicCredentialSources:
  """The ordered credential pool serving one user identity, best first.

  One credential is a pool of one. `user_scope` is the `risk_user_id` whose
  accounts lead the pool (empty for the org pool), `user_store_path` that
  user's store, and `store_path` the org store every pool ends with.
  """

  credentials: tuple[AnthropicCredential, ...]
  store_path: Path
  user_scope: str = ""
  user_store_path: Path | None = None

  @property
  def keys(self) -> tuple[str, ...]:
    return tuple(credential.key for credential in self.credentials)

  @property
  def tokens(self) -> tuple[str, ...]:
    return tuple(credential.token for credential in self.credentials)

  def by_key(self, key: str) -> AnthropicCredential | None:
    normalized = str(key or "").strip()
    for credential in self.credentials:
      if credential.key == normalized:
        return credential
    return None

  def by_token(self, token: str) -> AnthropicCredential | None:
    normalized = str(token or "").strip()
    for credential in self.credentials:
      if credential.token == normalized:
        return credential
    return None

  def by_generation_ref(self, generation_ref: str) -> AnthropicCredential | None:
    for credential in self.credentials:
      if credential.generation_ref == generation_ref:
        return credential
    return None


def _float_or_zero(value: Any) -> float:
  if isinstance(value, bool):
    return 0.0
  try:
    return float(value)
  except (TypeError, ValueError):
    return 0.0


def mask_anthropic_token(token: str) -> str:
  """A token rendered for an operator: enough to tell two apart, no secret."""

  text = str(token or "").strip()
  if not text:
    return "<empty>"
  if len(text) <= 16:
    return f"{text[:4]}…"
  return f"{text[:12]}…{text[-4:]}"


@contextlib.contextmanager
def _store_lock(path: Path) -> Iterator[None]:
  """Serialize store mutation across processes: login and refresh both write it."""

  try:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
  except OSError as exc:
    raise AnthropicStoreError(path, exc) from exc
  lock_path = path.with_name(f".{path.name}.lock")
  try:
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
  except OSError as exc:
    raise AnthropicStoreError(path, exc) from exc
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
def _store_read_lock(path: Path) -> Iterator[None]:
  """Take a shared lock on the store without creating anything.

  A read path (an exact `credential_source` resolve, `load`, `ensure_fresh`)
  must open the store read-only: no directory creation, no lock-file creation,
  no write lock. When the lock file the writers use is present, a shared lock
  keeps the read consistent with an in-flight writer; when it is absent (a
  read-only deployment that never wrote here) the read is a lock-free
  snapshot. A store directory this process cannot read is ``AnthropicStoreError``,
  never a raw ``OSError``.
  """

  lock_path = path.with_name(f".{path.name}.lock")
  try:
    fd = os.open(lock_path, os.O_RDONLY)
  except FileNotFoundError:
    yield
    return
  except OSError as exc:
    raise AnthropicStoreError(path, exc) from exc
  try:
    fcntl.flock(fd, fcntl.LOCK_SH)
    yield
  finally:
    try:
      fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
      os.close(fd)


@dataclass
class _AnthropicStore:
  accounts: list[AnthropicOAuthRecord] = field(default_factory=list)
  static_credentials: list[AnthropicOAuthRecord] = field(default_factory=list)
  retired_auth_token_grants: dict[str, str] = field(default_factory=dict)
  retirements: dict[str, str] = field(default_factory=dict)


def _grant_hash(record: AnthropicOAuthRecord) -> str:
  # Only enrollment/migration hashes material. Rotating OAuth tokens never
  # recompute the grant identity established by the original refresh grant.
  material = record.refresh_token or record.access_token
  return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _new_generation(record: AnthropicOAuthRecord) -> AnthropicOAuthRecord:
  return replace(
    record,
    grant_hash=_grant_hash(record),
    generation_id=secrets.token_hex(32),
  )


def _retire_generation(store: _AnthropicStore, record: AnthropicOAuthRecord, reason: str) -> None:
  if record.credential_kind == "auth_token" and record in store.static_credentials:
    store.retired_auth_token_grants[record.grant_hash] = record.generation_id
  store.retirements[record.generation_id] = reason


def _load_store_unlocked(path: Path, *, migrate: bool = True) -> _AnthropicStore:
  try:
    parsed = json.loads(path.read_text(encoding="utf-8"))
  except FileNotFoundError:
    return _AnthropicStore()
  except OSError as exc:
    raise AnthropicStoreError(path, exc) from exc
  except json.JSONDecodeError:
    raise AnthropicOAuthError("Invalid Anthropic credential store JSON") from None
  if not isinstance(parsed, dict):
    raise AnthropicOAuthError("Invalid Anthropic credential store payload")

  store = _AnthropicStore()
  retired_grants = parsed.get("retired_auth_token_grants")
  if isinstance(retired_grants, dict):
    store.retired_auth_token_grants = dict(retired_grants)
  retirements = parsed.get("retirements")
  if isinstance(retirements, dict):
    store.retirements = dict(retirements)
  for name, records in (("accounts", store.accounts), ("static_credentials", store.static_credentials)):
    entries = parsed.get(name)
    if isinstance(entries, list):
      for entry in entries:
        record = AnthropicOAuthRecord.from_payload(entry) if isinstance(entry, dict) else None
        if record is not None:
          records.append(record)
  if not isinstance(parsed.get("accounts"), list):
    legacy_token = str(parsed.get("auth_token") or "").strip()
    if legacy_token:
      store.accounts.append(AnthropicOAuthRecord(
        identity="legacy-setup-token",
        access_token=legacy_token,
        expires_at=_float_or_zero(parsed.get("expires_at")),
        created_at=_float_or_zero(parsed.get("created_at")),
        scope="",
        source=str(parsed.get("source") or "claude-setup-token").strip(),
      ))
  migrated = False
  if migrate:
    for records in (store.accounts, store.static_credentials):
      for index, record in enumerate(records):
        if not record.generation_id or not record.grant_hash:
          records[index] = _new_generation(record)
          migrated = True
    if migrated:
      _write_store_unlocked(path, store)
  return store


def load_anthropic_oauth_store(path: Path) -> tuple[AnthropicOAuthRecord, ...]:
  """Active pool accounts in enrollment order; migrate legacy/v2 once under lock.

  The read itself is a read-only snapshot (no store or lock-file creation).
  Migration is a write, so it is the only case that takes the write lock.
  """

  with _store_read_lock(path):
    store = _load_store_unlocked(path, migrate=False)
  if any(
    not record.generation_id or not record.grant_hash
    for records in (store.accounts, store.static_credentials)
    for record in records
  ):
    with _store_lock(path):
      store = _load_store_unlocked(path)
  return tuple(store.accounts)


def _write_store_unlocked(path: Path, store: _AnthropicStore) -> None:
  payload = {
    "version": ANTHROPIC_OAUTH_STORE_VERSION,
    "accounts": [record.to_payload() for record in store.accounts],
    "static_credentials": [record.to_payload() for record in store.static_credentials],
    "retired_auth_token_grants": store.retired_auth_token_grants,
    "retirements": store.retirements,
  }
  path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
  try:
    path.parent.chmod(0o700)
  except OSError:
    pass
  temp_path = path.with_name(f".{path.name}.{secrets.token_hex(12)}.tmp")
  try:
    fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
  except OSError as exc:
    raise AnthropicStoreError(path, exc) from exc
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
      json.dump(payload, handle, indent=2, sort_keys=True)
      handle.write("\n")
      handle.flush()
      os.fsync(handle.fileno())
    os.chmod(temp_path, 0o600)
    os.replace(temp_path, path)
    os.chmod(path, 0o600)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
  finally:
    try:
      temp_path.unlink()
    except FileNotFoundError:
      pass


def _enroll_record(
  store: _AnthropicStore,
  record: AnthropicOAuthRecord,
) -> AnthropicOAuthRecord:
  if not record.identity or not record.access_token:
    raise AnthropicOAuthError("Anthropic credential requires an identity and material")
  records = store.static_credentials if record.credential_kind == "api_key" else store.accounts
  for index, current in enumerate(records):
    if current.identity == record.identity:
      if not record.refresh_token and current.grant_hash == _grant_hash(record):
        return current
      enrolled = _new_generation(record)
      _retire_generation(store, current, "superseded")
      records[index] = enrolled
      return enrolled
  enrolled = _new_generation(record)
  records.append(enrolled)
  return enrolled


def _current_generation(store: _AnthropicStore, generation_id: str) -> AnthropicOAuthRecord:
  for record in (*store.accounts, *store.static_credentials):
    if record.generation_id == generation_id:
      return record
  raise AnthropicSourceError(store.retirements.get(generation_id, "missing"))


def upsert_anthropic_oauth_record(
  path: Path,
  record: AnthropicOAuthRecord,
) -> tuple[AnthropicOAuthRecord, ...]:
  """Enroll a grant, superseding same-account consent in its original pool slot.

  This is not a refresh update. Refresh owns its exact generation separately.
  Reobserving the same non-refreshing setup token retains its generation.
  """

  with _store_lock(path):
    store = _load_store_unlocked(path)
    _enroll_record(store, record)
    _write_store_unlocked(path, store)
    return tuple(store.accounts)


def remove_anthropic_oauth_record(path: Path, identity: str) -> bool:
  """Retire the account's current material, retaining secret-free source history."""

  normalized = str(identity or "").strip()
  with _store_lock(path):
    store = _load_store_unlocked(path)
    removed = False
    for records in (store.accounts, store.static_credentials):
      for record in tuple(records):
        if record.identity == normalized:
          _retire_generation(store, record, "removed")
          records.remove(record)
          removed = True
    if removed:
      _write_store_unlocked(path, store)
    return removed


def _pool_tokens(raw: str) -> tuple[str, ...]:
  """Sibling tokens carried by the JSON-array env form.

  Content this deployment wrote is reported and skipped rather than refused:
  a typo in the sibling list must not take the primary credential down with it.
  """

  text = str(raw or "").strip()
  if not text:
    return ()
  try:
    parsed = json.loads(text)
  except json.JSONDecodeError:
    LOGGER.warning(
      "%s is not valid JSON; continuing with the single-token credential pool",
      ANTHROPIC_AUTH_TOKENS_ENV,
    )
    return ()
  if not isinstance(parsed, list):
    LOGGER.warning(
      "%s must be a JSON array of setup tokens; continuing with the single-token pool",
      ANTHROPIC_AUTH_TOKENS_ENV,
    )
    return ()
  tokens: list[str] = []
  for entry in parsed:
    token = str(entry).strip() if isinstance(entry, str) else ""
    if token:
      tokens.append(token)
      continue
    LOGGER.warning(
      "%s entries must be non-empty token strings; skipping one entry",
      ANTHROPIC_AUTH_TOKENS_ENV,
    )
  return tuple(tokens)


def _single_token(raw: str) -> tuple[str, ...]:
  """The one token carried by a single-valued env form."""

  token = str(raw or "").strip()
  return (token,) if token else ()


# Pool membership as the environment carries it, in binding order. This tuple is
# what "the Anthropic pool" means for a process: the resolver below reads it,
# and every configuration surface that must put the same pool in front of
# another process of this product (the autonomous child environment projection)
# derives its names from `ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES` rather than
# listing credential names of its own.
_ENV_POOL_FORMS: tuple[tuple[str, Callable[[str], tuple[str, ...]]], ...] = (
  (ANTHROPIC_AUTH_TOKEN_ENV, _single_token),
  (ANTHROPIC_AUTH_TOKENS_ENV, _pool_tokens),
  (CLAUDE_CODE_OAUTH_TOKEN_ENV, _single_token),
)
ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES = frozenset(
  {name for name, _read in _ENV_POOL_FORMS}
  | {ANTHROPIC_AUTH_STORE_PATH_ENV, ANTHROPIC_STORE_BASE_ENV}
)


def _register_static_unlocked(
  store: _AnthropicStore,
  material: str,
  *,
  credential_kind: str,
  source: str,
) -> AnthropicOAuthRecord | None:
  current = next((
    record for record in store.static_credentials
    if record.credential_kind == credential_kind and record.source == source
  ), None)
  if not material:
    if current is not None:
      _retire_generation(store, current, "removed")
      store.static_credentials.remove(current)
    return None
  digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
  if current is not None and current.grant_hash == digest:
    return current
  # Opaque credentials have no remote account identity. This source-local
  # display identity follows setup_token_account_identity; the FULL hash and
  # random generation id, not this truncated label, identify the source grant.
  identity = f"api-key-{digest[:16]}" if credential_kind == "api_key" else setup_token_account_identity(material)
  moment = time.time()
  enrolled = _new_generation(AnthropicOAuthRecord(
    identity=identity,
    access_token=material,
    created_at=moment,
    updated_at=moment,
    scope="" if credential_kind == "api_key" else "user:inference",
    source=source,
    credential_kind=credential_kind,
  ))
  if current is not None:
    index = store.static_credentials.index(current)
    _retire_generation(store, current, "superseded")
    store.static_credentials[index] = enrolled
  else:
    store.static_credentials.append(enrolled)
  return enrolled


def register_anthropic_static_credential(
  material: str,
  *,
  credential_kind: str = "api_key",
  source: str = "env:ANTHROPIC_API_KEY",
  config: Mapping[str, Any] | None = None,
  environ: Mapping[str, str] | None = None,
  user_scope: Any = None,
) -> AnthropicCredential | None:
  """Register the effective opaque key/token in this existing source store.

  Source slots identify custody, not remote accounts: environment slots belong
  to the org; a request source (`_request_source`) belongs to the store of the
  scope that carried it. Same source/material retains its generation,
  replacement retires it, and empty material removes it. API keys and OAuth
  tokens retain their own credential kind within the pool.
  """

  path = resolve_anthropic_auth_store_path(
    config, environ=environ, user_scope="" if source.startswith("env:") else user_scope,
  )
  normalized = str(material or "").strip()
  with _store_lock(path):
    store = _load_store_unlocked(path)
    before = tuple(record.generation_id for record in store.static_credentials)
    record = _register_static_unlocked(
      store, normalized, credential_kind=credential_kind, source=source,
    )
    if before != tuple(current.generation_id for current in store.static_credentials):
      _write_store_unlocked(path, store)
    if record is None:
      return None
    return AnthropicCredential(
      key=record.identity, token=record.access_token, record=record, store_path=path,
    )


def _request_source(credential_kind: str, material: str) -> str:
  """A request-carried key/token's own source, named by its material's digest.

  Concurrent runs and users carry different material, so a request source is
  never a shared slot: one request's material cannot supersede another's, and
  the same material resolves its own generation again.
  """
  digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
  return f"request:anthropic:{credential_kind}:{digest}"


def _environment_source(name: str, material: str) -> str:
  """An environment token's own source, keyed by its material's digest.

  Every pool form keys by its own material, so two processes that share the
  org store but present different values for one name occupy distinct sources
  and never retire each other's generation. A name's slot is retired only when
  that name is observed empty (see `_sync_environment_tokens`).
  """

  digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
  return f"env:{name}:{digest}"


def enroll_request_anthropic_credential(
  material: str,
  *,
  credential_kind: str,
  user_scope: Any,
) -> AnthropicCredential | None:
  """Enroll supplied request material without deleting or re-owning an account."""
  normalized = str(material or "").strip()
  if not normalized:
    return None
  if credential_kind == "auth_token":
    sources = resolve_anthropic_credentials(user_scope=user_scope)
    enrolled = sources.by_token(normalized)
    if enrolled is not None:
      return enrolled
  return register_anthropic_static_credential(
    normalized,
    credential_kind=credential_kind,
    source=_request_source(credential_kind, normalized),
    user_scope=user_scope,
  )


def _environment_source_material(source: str, env: Mapping[str, str]) -> tuple[str, ...] | None:
  """Visible source material, or None when this process cannot observe the source."""

  name = source.removeprefix("env:").split(":", 1)[0]
  if name not in env:
    return None
  for env_name, read_form in _ENV_POOL_FORMS:
    if name == env_name:
      return read_form(env[name])
  return ()


def _sync_environment_tokens(
  path: Path,
  env: Mapping[str, str],
  enrolled_tokens: Mapping[str, tuple[Path, AnthropicOAuthRecord]],
) -> dict[tuple[str, str], AnthropicOAuthRecord]:
  """Reconcile observed org-env members; an absent name has no retirement authority.

  Each setup token's source is named by its own material's digest, exactly as a
  request-carried token is (`_request_source`). The deployment API key is
  registered separately by the gateway materializer. Two processes that share
  the org store but carry different setup values for a name therefore occupy
  distinct sources: neither retires the other's generation.
  A process retires only the members of a source it actually observes to have
  changed or emptied — a name present-but-empty still retires that name's slot
  (per the reference: an emptied env name retires its slot), while an absent
  name asserts nothing.
  """

  selected: dict[tuple[str, str], AnthropicOAuthRecord] = {}
  with _store_lock(path):
    store = _load_store_unlocked(path)
    before = tuple(record.generation_id for record in store.static_credentials)
    desired: set[str] = set()
    for name, read_form in _ENV_POOL_FORMS:
      for token in read_form(env.get(name, "")):
        if token in enrolled_tokens:
          continue
        source = _environment_source(name, token)
        desired.add(source)
        record = _register_static_unlocked(
          store, token, credential_kind="auth_token", source=source,
        )
        if record is not None:
          selected[(name, token)] = record
    pool_names = {name for name, _read in _ENV_POOL_FORMS}
    for record in tuple(store.static_credentials):
      if not record.source.startswith("env:"):
        continue
      name = record.source.removeprefix("env:").split(":", 1)[0]
      # A name this process cannot observe asserts nothing.
      if name not in pool_names or name not in env:
        continue
      if name == ANTHROPIC_AUTH_TOKENS_ENV:
        # Membership reconciliation: an array member this observation no longer
        # names is retired as removed.
        if record.source not in desired:
          _retire_generation(store, record, "removed")
          store.static_credentials.remove(record)
      elif _environment_source_material(record.source, env) == ():
        # A single-valued name retires its slot only when observed empty: a
        # different non-empty value is a co-existing source, not a retirement.
        _retire_generation(store, record, "removed")
        store.static_credentials.remove(record)
    if before != tuple(record.generation_id for record in store.static_credentials):
      _write_store_unlocked(path, store)
  return selected


def resolve_anthropic_credentials(
  config: Mapping[str, Any] | None = None,
  *,
  environ: Mapping[str, str] | None = None,
  user_scope: Any = None,
) -> AnthropicCredentialSources:
  """The pool bound to one user identity, highest precedence first.

  One ordered derivation: the credential this request already carries, the
  accounts enrolled for this user, the deployment API key explicitly enrolled
  by the gateway, then org credentials (environment setup tokens, then accounts
  enrolled for the deployment). Exact generation lookup remains read-only and
  serves only the member that the request already bound. Ambient API keys are
  never a source for this resolver.

  An environment token that happens to be an enrolled account's current access
  token resolves to that account, so it is refreshable and parks under the
  account's identity rather than twice under two keys.
  """

  cfg = config or {}
  env = os.environ if environ is None else environ
  deployment_key_ref = str(cfg.get(ANTHROPIC_DEPLOYMENT_API_KEY_FIELD) or "")
  scope = anthropic_user_scope(
    user_scope if user_scope is not None else cfg.get(ANTHROPIC_USER_SCOPE_FIELD)
  )
  org_path = resolve_anthropic_auth_store_path(cfg, environ=env, user_scope="")
  user_path = (
    resolve_anthropic_auth_store_path(cfg, environ=env, user_scope=scope) if scope else None
  )
  store_paths = (org_path,) if user_path is None else (user_path, org_path)
  bound_source = cfg.get("credential_source")
  if isinstance(bound_source, Mapping):
    # The documented read-only exact lookup. A bound `credential_source` names
    # one immutable generation: read it under the store's read lock without
    # creating the store, without an env sync, and without registering or
    # retiring anything. A retired/removed/superseded generation is the typed
    # source error with its recorded reason; a changed env source is reported
    # here too, never rewritten. Reconciliation is the pool path below.
    credential = _resolve_generation(
      str(bound_source.get("generation_ref") or ""),
      environ=env, user_scope=scope, store_paths=store_paths,
      deployment_api_key_source=deployment_key_ref,
    )
    return AnthropicCredentialSources(
      credentials=(credential,),
      store_path=org_path,
      user_scope=scope,
      user_store_path=user_path,
    )
  api_credential = None
  if deployment_key_ref:
    # Membership reads active records only. A retired key no longer belongs to
    # the pool, but does not prevent the remaining seats from serving rotation.
    with _store_read_lock(org_path):
      org_store = _load_store_unlocked(org_path, migrate=False)
      deployment_key = next((
        record for record in org_store.static_credentials
        if record.generation_ref == deployment_key_ref
        and record.source == "env:ANTHROPIC_API_KEY"
        and record.credential_kind == "api_key"
      ), None)
    if deployment_key is not None:
      api_credential = AnthropicCredential(
        key=deployment_key.identity, token=deployment_key.access_token,
        record=deployment_key, store_path=org_path,
      )
  stores = tuple((path, load_anthropic_oauth_store(path)) for path in store_paths)
  # The user's own record wins when the same access token sits in both stores:
  # it names the account the request is entitled to refresh and park under.
  by_access_token = {
    record.access_token: (path, record)
    for path, records in reversed(stores)
    for record in records
  }
  env_records = _sync_environment_tokens(
    org_path, env,
    {record.access_token: (org_path, record) for record in stores[-1][1]},
  )

  credentials: list[AnthropicCredential] = []
  seen_keys: set[str] = set()

  def _append(credential: AnthropicCredential) -> None:
    if credential.key in seen_keys:
      return
    seen_keys.add(credential.key)
    credentials.append(credential)

  def _append_token(token: str, *, name: str = "") -> None:
    stored = by_access_token.get(token)
    if stored is None:
      if name:
        record = env_records[(name, token)]
        credential = AnthropicCredential(
          key=token, token=record.access_token, record=record, store_path=org_path,
        )
      else:
        # Re-entry of an already owned static source preserves its provenance.
        # Full static hashes also identify retired material without retaining
        # old secrets; that is not consent to resurrect it as a request source.
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        for path in store_paths:
          with _store_lock(path):
            store = _load_store_unlocked(path)
            current = next((
              record for record in store.static_credentials
              if record.credential_kind == "auth_token" and record.access_token == token
            ), None)
            retired_id = store.retired_auth_token_grants.get(digest, "")
            known_ref = current.generation_ref if current is not None else (
              f"{_GENERATION_REF_PREFIX}{retired_id}" if retired_id else ""
            )
          if known_ref:
            credential = _resolve_generation(
              known_ref, environ=env, user_scope=scope, store_paths=store_paths,
            )
            _append(credential)
            return
        credential = register_anthropic_static_credential(
          token, credential_kind="auth_token", source=_request_source("auth_token", token),
          config=cfg, environ=env, user_scope=scope,
        )
        if credential is None:
          return
        credential = replace(credential, key=token)
      _append(credential)
      return
    path, record = stored
    _append(AnthropicCredential(
      key=record.identity,
      token=record.access_token,
      record=record,
      store_path=path,
    ))

  def _append_records(path: Path, records: Sequence[AnthropicOAuthRecord]) -> None:
    for record in records:
      _append(AnthropicCredential(
        key=record.identity,
        token=record.access_token,
        record=record,
        store_path=path,
      ))

  bound = str(cfg.get("auth_token") or "").strip()
  if bound:
    _append_token(bound)
  for path, records in stores[:-1]:
    _append_records(path, records)
  if api_credential is not None:
    _append(api_credential)
  for name, read_form in _ENV_POOL_FORMS:
    for token in read_form(env.get(name, "")):
      _append_token(token, name=name)
  _append_records(*stores[-1])

  return AnthropicCredentialSources(
    credentials=tuple(credentials),
    store_path=org_path,
    user_scope=scope,
    user_store_path=user_path,
  )


def _resolve_generation(
  generation_ref: str,
  *,
  environ: Mapping[str, str] | None,
  user_scope: Any,
  store_paths: Sequence[Path] | None = None,
  deployment_api_key_source: str = "",
) -> AnthropicCredential:
  scope = anthropic_user_scope(user_scope)
  org_path = resolve_anthropic_auth_store_path(environ=environ, user_scope="")
  paths = store_paths if store_paths is not None else (
    (resolve_anthropic_auth_store_path(environ=environ, user_scope=scope), org_path)
    if scope else (org_path,)
  )
  generation_id = generation_ref.removeprefix(_GENERATION_REF_PREFIX)
  for path in paths:
    with _store_read_lock(path):
      store = _load_store_unlocked(path, migrate=False)
      try:
        record = _current_generation(store, generation_id)
      except AnthropicSourceError as exc:
        if exc.reason != "missing":
          raise
        continue
      if record.source == "env:ANTHROPIC_API_KEY":
        if record.generation_ref != deployment_api_key_source:
          raise AnthropicSourceError("unobservable")
      elif record.source.startswith("env:"):
        env = os.environ if environ is None else environ
        available = _environment_source_material(record.source, env)
        if available is None:
          raise AnthropicSourceError("unobservable")
        if record.access_token not in available:
          raise AnthropicSourceError("superseded" if available else "removed")
      key = (
        record.identity
        if record in store.accounts or record.credential_kind == "api_key"
        else record.access_token
      )
      return AnthropicCredential(
        key=key, token=record.access_token, record=record, store_path=path,
      )
  raise AnthropicSourceError("missing")


def is_anthropic_generation_ref(ref: str) -> bool:
  """Whether a credential reference names an Anthropic source generation."""
  return ref.startswith(_GENERATION_REF_PREFIX)


def anthropic_generation_auth_config(
  generation_ref: str,
  *,
  user_scope: Any,
  base_config: Mapping[str, Any],
) -> dict[str, Any]:
  """Materialize an exact active source generation without pool selection."""
  credential = _resolve_generation(
    generation_ref, environ=None, user_scope=user_scope,
    deployment_api_key_source=str(base_config.get(ANTHROPIC_DEPLOYMENT_API_KEY_FIELD) or ""),
  )
  assert credential.record is not None
  is_api_key = credential.record.credential_kind == "api_key"
  config = dict(base_config)
  config.update(
    provider="anthropic",
    auth_mode="api" if is_api_key else "oauth",
    api_key=credential.token if is_api_key else "",
    auth_token="" if is_api_key else credential.token,
    credential_source=credential.source_metadata(),
    auth_user_scope=anthropic_user_scope(user_scope),
  )
  return config


class AnthropicCredentialPool(ProviderCredentialPool):
  """Which sibling credential is usable right now, for this process.

  The resolver owns the order; the limiter's own reset headers own the block
  windows. Selection is a read of those two and never a decision of its own.
  Keys are credential keys (an enrolled account's identity, or an environment
  token), so a refreshed access token stays parked for its account's window.
  """



# Usage limits outlive a single request and a single session, so the block state
# is process-wide: a credential the limiter rejected for the next five hours must
# not be handed to the next session that starts.
ANTHROPIC_CREDENTIAL_POOL = AnthropicCredentialPool()
_PROVIDER_POOLS["anthropic"] = ANTHROPIC_CREDENTIAL_POOL
_LOGGED_POOL_SUMMARIES: dict[str, str] = {}


def _post_token_endpoint(url: str, body: JSONMapping) -> JSONMapping:
  import httpx

  try:
    response = httpx.post(
      url,
      json=dict(body),
      headers={"Content-Type": "application/json", "Accept": "application/json"},
      timeout=httpx.Timeout(HTTP_TIMEOUT_SECONDS, connect=HTTP_CONNECT_TIMEOUT_SECONDS),
    )
  except httpx.HTTPError:
    raise AnthropicOAuthError("Anthropic OAuth token request failed") from None
  if response.status_code >= 400:
    raise AnthropicOAuthError(
      f"Anthropic OAuth token request failed ({response.status_code})"
    )
  try:
    payload = response.json()
  except ValueError as exc:
    raise AnthropicOAuthError("Anthropic OAuth token response was not valid JSON") from exc
  if not isinstance(payload, dict):
    raise AnthropicOAuthError("Anthropic OAuth token response was not a JSON object")
  return payload


def _get_profile_endpoint(url: str, access_token: str) -> JSONMapping:
  import httpx

  try:
    response = httpx.get(
      url,
      headers={
        "Authorization": f"Bearer {access_token}",
        "anthropic-beta": ANTHROPIC_OAUTH_BETA,
        "Accept": "application/json",
      },
      timeout=httpx.Timeout(HTTP_TIMEOUT_SECONDS, connect=HTTP_CONNECT_TIMEOUT_SECONDS),
    )
  except httpx.HTTPError:
    raise AnthropicOAuthError("Anthropic OAuth profile request failed") from None
  if response.status_code >= 400:
    raise AnthropicOAuthError(
      f"Anthropic OAuth profile request failed ({response.status_code})"
    )
  try:
    payload = response.json()
  except ValueError as exc:
    raise AnthropicOAuthError("Anthropic OAuth profile response was not valid JSON") from exc
  if not isinstance(payload, dict):
    raise AnthropicOAuthError("Anthropic OAuth profile response was not a JSON object")
  return payload


def _token_payload_fields(payload: JSONMapping) -> tuple[str, str, float, str]:
  access_token = str(payload.get("access_token") or "").strip()
  if not access_token:
    raise AnthropicOAuthError("Anthropic OAuth token response carried no access_token")
  refresh_token = str(payload.get("refresh_token") or "").strip()
  expires_in = _float_or_zero(payload.get("expires_in"))
  scope = str(payload.get("scope") or ANTHROPIC_OAUTH_SCOPE).strip()
  return access_token, refresh_token, expires_in, scope


def _identity_from_payload(payload: JSONMapping) -> str:
  account = payload.get("account")
  if isinstance(account, Mapping):
    for field in ("email_address", "email", "uuid", "id", "account_uuid"):
      value = str(account.get(field) or "").strip()
      if value:
        return value
  for field in ("account_email", "email"):
    value = str(payload.get(field) or "").strip()
    if value:
      return value
  organization = payload.get("organization")
  if isinstance(organization, Mapping):
    value = str(organization.get("uuid") or "").strip()
    if value:
      return value
  return ""


def refresh_anthropic_oauth_record(
  record: AnthropicOAuthRecord,
  *,
  store_path: Path,
  now: float | None = None,
  post: TokenPoster | None = None,
) -> AnthropicOAuthRecord:
  """Refresh the exact current grant under its owner's cross-process lock.

  Reloading before the exchange prevents reuse of a rotating refresh token.
  Consent/removal and the entire refresh exchange+save share the same lock.
  Captured material from a retired generation can never overwrite its successor.
  """

  moment = time.time() if now is None else now
  with _store_lock(store_path):
    store = _load_store_unlocked(store_path)
    current = _current_generation(store, record.generation_id)
    if current.access_token != record.access_token or current.refresh_token != record.refresh_token:
      return current
    return _refresh_record_unlocked(store_path, store, current, moment=moment, post=post)


def _refresh_record_unlocked(
  path: Path,
  store: _AnthropicStore,
  record: AnthropicOAuthRecord,
  *,
  moment: float,
  post: TokenPoster | None = None,
) -> AnthropicOAuthRecord:
  if not record.needs_refresh(now=moment):
    return record
  poster = post or _post_token_endpoint
  try:
    payload = poster(
      ANTHROPIC_OAUTH_TOKEN_URL,
      {
        "grant_type": "refresh_token",
        "refresh_token": record.refresh_token,
        "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
      },
    )
    access_token, refresh_token, expires_in, scope = _token_payload_fields(payload)
  except AnthropicOAuthError:
    failed = replace(record, last_error="Anthropic OAuth refresh failed", last_error_at=moment)
    store.accounts[store.accounts.index(record)] = failed
    _write_store_unlocked(path, store)
    raise AnthropicOAuthError("Anthropic OAuth refresh failed") from None
  refreshed = replace(
    record,
    access_token=access_token,
    refresh_token=refresh_token or record.refresh_token,
    expires_at=moment + expires_in if expires_in else moment,
    updated_at=moment,
    scope=scope or record.scope,
    last_error="",
    last_error_at=0.0,
  )
  store.accounts[store.accounts.index(record)] = refreshed
  _write_store_unlocked(path, store)
  LOGGER.info(
    "Refreshed Anthropic OAuth access token for %s (expires in %.0fs)",
    refreshed.identity,
    max(0.0, refreshed.expires_at - moment),
  )
  return refreshed


def _current_credential(
  credential: AnthropicCredential,
  *,
  refresh: bool,
  now: float | None = None,
  post: TokenPoster | None = None,
) -> AnthropicCredential:
  if credential.record is None or credential.store_path is None:
    raise AnthropicSourceError("unregistered")
  path = credential.store_path
  with _store_lock(path):
    store = _load_store_unlocked(path)
    record = _current_generation(store, credential.generation_id)
    if refresh:
      record = _refresh_record_unlocked(
        path, store, record, moment=time.time() if now is None else now, post=post,
      )
    return replace(credential, token=record.access_token, record=record)


def _usable_credential(
  candidates: Sequence[AnthropicCredential],
  *,
  pool: AnthropicCredentialPool,
  now: float | None = None,
) -> AnthropicCredential | None:
  """The first unparked source, with material and metadata from the same generation."""

  moment = time.time() if now is None else now
  for credential in candidates:
    if pool.blocked_until(credential.key) > moment:
      continue
    try:
      return _current_credential(credential, refresh=True, now=moment)
    except AnthropicOAuthError as exc:
      # Keep the existing account-level refresh-failure parking policy.
      pool.block(credential.key, until=moment + REFRESH_FAILURE_BLOCK_SECONDS)
      LOGGER.warning(
        "Anthropic OAuth refresh failed for %s; parking it for %.0fs: %s",
        credential.account_identity,
        REFRESH_FAILURE_BLOCK_SECONDS,
        exc,
      )
  return None


def _log_pool_summary(sources: AnthropicCredentialSources) -> None:
  summary = ", ".join(
    f"{credential.account_identity}"
    f" generation={credential.generation_id}"
    for credential in sources.credentials
  )
  if _LOGGED_POOL_SUMMARIES.get(sources.user_scope) == summary:
    return
  _LOGGED_POOL_SUMMARIES[sources.user_scope] = summary
  LOGGER.info(
    "Anthropic credential pool for %s: %d credential(s) [%s]; user_store=%s; org_store=%s",
    f"user {sources.user_scope}" if sources.user_scope else "the org (no user pool)",
    len(sources.credentials),
    summary or "none",
    sources.user_store_path or "none",
    sources.store_path,
  )


def select_anthropic_credential_source(
  sources: AnthropicCredentialSources,
  *,
  pool: AnthropicCredentialPool | None = None,
  now: float | None = None,
) -> AnthropicCredential | None:
  """Bind a credential: the first usable one, else the first at all.

  A pool of one is never withheld from a run — an expired or unrefreshable
  single credential still binds, so the turn walks into the provider's existing
  rate-limit/auth handling instead of failing here with no request made.
  """

  selected_pool = ANTHROPIC_CREDENTIAL_POOL if pool is None else pool
  _log_pool_summary(sources)
  selected = _usable_credential(sources.credentials, pool=selected_pool, now=now)
  if selected is not None:
    return selected
  return _current_credential(sources.credentials[0], refresh=False) if sources.credentials else None


def rotate_anthropic_credential_source(
  sources: AnthropicCredentialSources,
  *,
  current: str,
  until: float,
  pool: AnthropicCredentialPool | None = None,
  now: float | None = None,
) -> AnthropicCredential | None:
  """Park the current credential until `until` and hand back the next sibling.

  Unlike binding there is no fallback here: handing back a credential a limiter
  already rejected would rotate the run in circles instead of letting the
  caller's existing retry run its course.
  """

  selected_pool = ANTHROPIC_CREDENTIAL_POOL if pool is None else pool
  normalized = str(current or "").strip()
  current_credential = sources.by_token(normalized)
  current_key = current_credential.key if current_credential is not None else normalized
  selected_pool.block(current_key, until=until)
  siblings = [
    credential for credential in sources.credentials if credential.key != current_key
  ]
  return _usable_credential(siblings, pool=selected_pool, now=now)


def ensure_fresh_anthropic_credential(
  token: str,
  *,
  config: Mapping[str, Any] | None = None,
  environ: Mapping[str, str] | None = None,
  now: float | None = None,
) -> str:
  """The access token to send for `token`, refreshed first when it is stale.

  A non-refreshing setup/environment source is sent as-is. Supplied source
  metadata resolves its exact original generation even if `token` predates
  a refresh; it never falls back to the pool's current access-token match.
  Owned OAuth grants refresh at this credential-to-client boundary.
  """

  normalized = str(token or "").strip()
  bound_source = (config or {}).get("credential_source")
  if not normalized and not isinstance(bound_source, Mapping):
    return ""
  sources = resolve_anthropic_credentials(config, environ=environ)
  if isinstance(bound_source, Mapping):
    credential = sources.by_generation_ref(str(bound_source.get("generation_ref") or ""))
    if credential is None:
      raise AnthropicSourceError("missing")
  else:
    credential = sources.by_token(normalized)
  if credential is None or credential.record is None or credential.store_path is None:
    return normalized
  record = credential.record
  if not record.needs_refresh(now=now):
    return credential.token
  try:
    return refresh_anthropic_oauth_record(
      record,
      store_path=credential.store_path,
      now=now,
    ).access_token
  except AnthropicSourceError:
    raise
  except AnthropicOAuthError as exc:
    # The bound credential is still handed to the request: the provider's 401
    # is what rotates the run to a sibling, and withholding it here would end
    # the turn with no request made at all.
    LOGGER.warning("Anthropic OAuth refresh failed for %s: %s", record.identity, exc)
    return credential.token


@dataclass(frozen=True)
class AnthropicLoginRequest:
  """The browser step of one enrollment: what to open, and what must come back."""

  authorize_url: str
  state: str
  code_verifier: str


def generate_pkce() -> tuple[str, str]:
  verifier_bytes = secrets.token_bytes(32)
  verifier = base64.urlsafe_b64encode(verifier_bytes).rstrip(b"=").decode("ascii")
  challenge_bytes = hashlib.sha256(verifier.encode("ascii")).digest()
  challenge = base64.urlsafe_b64encode(challenge_bytes).rstrip(b"=").decode("ascii")
  return verifier, challenge


def start_anthropic_login() -> AnthropicLoginRequest:
  verifier, challenge = generate_pkce()
  state = base64.urlsafe_b64encode(secrets.token_bytes(24)).rstrip(b"=").decode("ascii")
  params = {
    "code": "true",
    "response_type": "code",
    "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
    "redirect_uri": ANTHROPIC_OAUTH_REDIRECT_URI,
    "scope": ANTHROPIC_OAUTH_SCOPE,
    "code_challenge": challenge,
    "code_challenge_method": "S256",
    "state": state,
  }
  authorize_url = f"{ANTHROPIC_OAUTH_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"
  return AnthropicLoginRequest(authorize_url=authorize_url, state=state, code_verifier=verifier)


def parse_anthropic_callback(raw: str) -> tuple[str, str]:
  """Extract code and state from a callback URL, query string, or `code#state`."""

  value = str(raw or "").strip()
  if not value:
    return "", ""
  if value.startswith(("http://", "https://")):
    parsed = urllib.parse.urlparse(value)
    params = urllib.parse.parse_qs(parsed.query)
    return params.get("code", [""])[0].strip(), params.get("state", [""])[0].strip()
  query = value.lstrip("?")
  if "code=" in query:
    params = urllib.parse.parse_qs(query)
    return params.get("code", [""])[0].strip(), params.get("state", [""])[0].strip()
  if "#" in value:
    code, state = value.split("#", 1)
    return code.strip(), state.strip()
  return value, ""


def complete_anthropic_login(
  request: AnthropicLoginRequest,
  pasted: str,
  *,
  store_path: Path,
  now: float | None = None,
  post: TokenPoster | None = None,
  get_profile: ProfileGetter | None = None,
) -> AnthropicOAuthRecord:
  """Exchange the pasted authorization code and enroll the account it names.

  The returned `state` is compared with the one this process generated: an
  authorization code that arrived bound to a different request is a
  cross-request substitution and is refused (the single security invariant this
  flow enforces locally).
  """

  code, state = parse_anthropic_callback(pasted)
  if not code:
    raise AnthropicOAuthError("No authorization code found in the pasted callback value")
  if state and state != request.state:
    raise AnthropicOAuthError(
      "The pasted callback state does not match this login request; start the login again"
    )
  moment = time.time() if now is None else now
  poster = post or _post_token_endpoint
  payload = poster(
    ANTHROPIC_OAUTH_TOKEN_URL,
    {
      "grant_type": "authorization_code",
      "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
      "code": code,
      "code_verifier": request.code_verifier,
      "redirect_uri": ANTHROPIC_OAUTH_REDIRECT_URI,
      "state": request.state,
    },
  )
  access_token, refresh_token, expires_in, scope = _token_payload_fields(payload)
  if not refresh_token:
    raise AnthropicOAuthError(
      "Anthropic OAuth login returned no refresh token; the gateway cannot keep this "
      "account enrolled without one"
    )
  identity = _identity_from_payload(payload)
  if not identity:
    profile_getter = get_profile or _get_profile_endpoint
    identity = _identity_from_payload(profile_getter(ANTHROPIC_OAUTH_PROFILE_URL, access_token))
  if not identity:
    raise AnthropicOAuthError(
      "Anthropic OAuth login did not report an account identity (email or account id); "
      "the account cannot be enrolled without one"
    )
  record = AnthropicOAuthRecord(
    identity=identity,
    access_token=access_token,
    refresh_token=refresh_token,
    expires_at=moment + expires_in if expires_in else moment,
    created_at=moment,
    updated_at=moment,
    scope=scope,
    source=ANTHROPIC_OAUTH_SOURCE,
  )
  enrolled = upsert_anthropic_oauth_record(store_path, record)
  return next(current for current in enrolled if current.identity == identity)


def setup_token_account_identity(token: str) -> str:
  """The account key for a `claude setup-token` credential.

  A setup token carries `user:inference` only, so Anthropic's profile endpoint
  refuses to name the account it belongs to (403 `oauth_scope_insufficient`).
  The account key is therefore derived from the token itself: stable across
  re-enrollments of the same token, printable (it is not the token), and the
  key the process-wide block state uses — so one usage-limited account is
  parked for every user who enrolled it.
  """

  digest = hashlib.sha256(str(token or "").strip().encode("utf-8")).hexdigest()[:16]
  return f"setup-token-{digest}"


def enroll_anthropic_setup_token(
  token: str,
  *,
  store_path: Path,
  now: float | None = None,
) -> AnthropicOAuthRecord:
  """Enroll an account whose credential is a `claude setup-token` value.

  Same pool membership as a login-enrolled account, minus the refresh grant a
  setup token does not have: it is sent as-is, exactly like the environment
  forms, and it is parked by its account key when a limiter rejects it.
  """

  normalized = str(token or "").strip()
  if not normalized:
    raise AnthropicOAuthError("No Anthropic setup token was supplied")
  moment = time.time() if now is None else now
  record = AnthropicOAuthRecord(
    identity=setup_token_account_identity(normalized),
    access_token=normalized,
    refresh_token="",
    expires_at=0.0,
    created_at=moment,
    updated_at=moment,
    scope="user:inference",
    source="claude-setup-token",
  )
  enrolled = upsert_anthropic_oauth_record(store_path, record)
  return next(current for current in enrolled if current.identity == record.identity)



def anthropic_token_store_is_private(path: Path) -> bool:
  try:
    return stat.S_IMODE(path.stat().st_mode) == 0o600
  except OSError:
    return False


__all__ = [
  "ANTHROPIC_AUTH_STORE_PATH_ENV",
  "ANTHROPIC_AUTH_TOKEN_ENV",
  "ANTHROPIC_AUTH_TOKENS_ENV",
  "ANTHROPIC_CREDENTIAL_POOL",
  "ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES",
  "ANTHROPIC_DEPLOYMENT_API_KEY_FIELD",
  "ANTHROPIC_OAUTH_AUTHORIZE_URL",
  "ANTHROPIC_OAUTH_CLIENT_ID",
  "ANTHROPIC_OAUTH_REDIRECT_URI",
  "ANTHROPIC_OAUTH_SCOPE",
  "ANTHROPIC_OAUTH_TOKEN_URL",
  "ANTHROPIC_STORE_BASE_ENV",
  "ANTHROPIC_USER_SCOPE_FIELD",
  "CLAUDE_CODE_OAUTH_TOKEN_ENV",
  "AnthropicCredential",
  "AnthropicCredentialPool",
  "AnthropicCredentialSources",
  "AnthropicLoginRequest",
  "AnthropicOAuthError",
  "AnthropicOAuthRecord",
  "AnthropicSourceError",
  "REFRESH_FAILURE_BLOCK_SECONDS",
  "REFRESH_MARGIN_SECONDS",
  "UNDECLARED_LIMIT_BLOCK_SECONDS",
  "anthropic_generation_auth_config",
  "is_anthropic_generation_ref",
  "anthropic_token_store_is_private",
  "anthropic_user_scope",
  "generate_pkce",
  "complete_anthropic_login",
  "enroll_anthropic_setup_token",
  "enroll_request_anthropic_credential",
  "ensure_fresh_anthropic_credential",
  "load_anthropic_oauth_store",
  "mask_anthropic_token",
  "parse_anthropic_callback",
  "refresh_anthropic_oauth_record",
  "register_anthropic_static_credential",
  "remove_anthropic_oauth_record",
  "resolve_anthropic_auth_store_path",
  "resolve_anthropic_credentials",
  "rotate_anthropic_credential_source",
  "select_anthropic_credential_source",
  "setup_token_account_identity",
  "start_anthropic_login",
  "upsert_anthropic_oauth_record",
]
