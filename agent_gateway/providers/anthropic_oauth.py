from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import json
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
import secrets
import stat
import time
import urllib.parse
from typing import Any, Callable, Iterator, Mapping, Sequence


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
ANTHROPIC_OAUTH_STORE_VERSION = 2

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
  """One enrolled Claude account: its own tokens, its own expiry, its own errors.

  `identity` is the account the grant belongs to (email address, else account
  id). It is the record's key: it survives refresh, so parking an account for a
  usage window is not undone by that account's access token rotating.
  """

  identity: str
  access_token: str
  refresh_token: str = ""
  expires_at: float = 0.0
  created_at: float = 0.0
  updated_at: float = 0.0
  scope: str = ANTHROPIC_OAUTH_SCOPE
  source: str = ANTHROPIC_OAUTH_SOURCE
  last_error: str = ""
  last_error_at: float = 0.0

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
    )


@dataclass(frozen=True)
class AnthropicCredential:
  """A pool member: what to send, what to park, and who can refresh it.

  `key` is what the process-wide block state is keyed by — an enrolled
  account's identity, or the token itself for a credential supplied through the
  environment, which has no identity and no refresh token. `store_path` is the
  store the record was read from, so a refresh is written back to the store
  that owns the account rather than to whichever store the pool started at.
  """

  key: str
  token: str
  record: AnthropicOAuthRecord | None = None
  store_path: Path | None = None


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

  path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
  lock_path = path.with_name(f".{path.name}.lock")
  fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
  try:
    fcntl.flock(fd, fcntl.LOCK_EX)
    yield
  finally:
    try:
      fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
      os.close(fd)


def load_anthropic_oauth_store(path: Path) -> tuple[AnthropicOAuthRecord, ...]:
  """Every enrolled account, in enrollment order.

  A store written before this product had its own login holds a single
  setup-token record; it is read as one account with no refresh token, so an
  existing deployment keeps serving from it and nothing has to be re-imported.
  """

  try:
    raw = path.read_text(encoding="utf-8")
  except FileNotFoundError:
    return ()
  except OSError as exc:
    raise AnthropicOAuthError(f"Unable to read Anthropic OAuth token store: {path}") from exc
  try:
    parsed = json.loads(raw)
  except json.JSONDecodeError as exc:
    raise AnthropicOAuthError(f"Invalid Anthropic OAuth token store JSON: {path}") from exc
  if not isinstance(parsed, dict):
    raise AnthropicOAuthError(f"Invalid Anthropic OAuth token store payload: {path}")

  accounts = parsed.get("accounts")
  if isinstance(accounts, list):
    records = [AnthropicOAuthRecord.from_payload(entry) for entry in accounts if isinstance(entry, dict)]
    return tuple(record for record in records if record is not None)

  legacy_token = str(parsed.get("auth_token") or "").strip()
  if not legacy_token:
    return ()
  return (
    AnthropicOAuthRecord(
      identity="legacy-setup-token",
      access_token=legacy_token,
      expires_at=_float_or_zero(parsed.get("expires_at")),
      created_at=_float_or_zero(parsed.get("created_at")),
      scope="",
      source=str(parsed.get("source") or "claude-setup-token").strip(),
    ),
  )


def save_anthropic_oauth_store(path: Path, records: Sequence[AnthropicOAuthRecord]) -> None:
  """Replace the store atomically at mode 0600, preserving enrollment order."""

  payload = {
    "version": ANTHROPIC_OAUTH_STORE_VERSION,
    "accounts": [record.to_payload() for record in records],
  }
  path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
  try:
    path.parent.chmod(0o700)
  except OSError:
    pass
  temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
  fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
      json.dump(payload, handle, indent=2, sort_keys=True)
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


def upsert_anthropic_oauth_record(
  path: Path,
  record: AnthropicOAuthRecord,
) -> tuple[AnthropicOAuthRecord, ...]:
  """Write one account, keyed by identity, without disturbing its siblings.

  The read happens under the same lock as the write: a refresh landing while
  another account is being enrolled must not drop either one.
  """

  if not record.identity or not record.access_token:
    raise AnthropicOAuthError("Anthropic OAuth record requires an identity and an access token")
  with _store_lock(path):
    existing = list(load_anthropic_oauth_store(path))
    for index, current in enumerate(existing):
      if current.identity == record.identity:
        existing[index] = record
        break
    else:
      existing.append(record)
    save_anthropic_oauth_store(path, existing)
    return tuple(existing)


def remove_anthropic_oauth_record(path: Path, identity: str) -> bool:
  normalized = str(identity or "").strip()
  with _store_lock(path):
    existing = list(load_anthropic_oauth_store(path))
    remaining = [record for record in existing if record.identity != normalized]
    if len(remaining) == len(existing):
      return False
    save_anthropic_oauth_store(path, remaining)
    return True


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


def resolve_anthropic_credentials(
  config: Mapping[str, Any] | None = None,
  *,
  environ: Mapping[str, str] | None = None,
  user_scope: Any = None,
) -> AnthropicCredentialSources:
  """The pool bound to one user identity, highest precedence first.

  One ordered derivation, not a choice between two modes: the credential this
  request already carries, then the accounts enrolled for this user, then the
  org credentials (the environment forms, then the accounts enrolled for the
  deployment). A user who has enrolled nothing therefore resolves exactly the
  org pool, and a user who has enrolled accounts spends those first.

  An environment token that happens to be an enrolled account's current access
  token resolves to that account, so it is refreshable and parks under the
  account's identity rather than twice under two keys.
  """

  cfg = config or {}
  env = os.environ if environ is None else environ
  scope = anthropic_user_scope(
    user_scope if user_scope is not None else cfg.get(ANTHROPIC_USER_SCOPE_FIELD)
  )
  org_path = resolve_anthropic_auth_store_path(cfg, environ=env, user_scope="")
  user_path = (
    resolve_anthropic_auth_store_path(cfg, environ=env, user_scope=scope) if scope else None
  )
  store_paths = (org_path,) if user_path is None else (user_path, org_path)
  stores = tuple((path, load_anthropic_oauth_store(path)) for path in store_paths)
  # The user's own record wins when the same access token sits in both stores:
  # it names the account the request is entitled to refresh and park under.
  by_access_token = {
    record.access_token: (path, record)
    for path, records in reversed(stores)
    for record in records
  }

  credentials: list[AnthropicCredential] = []
  seen_keys: set[str] = set()

  def _append(credential: AnthropicCredential) -> None:
    if credential.key in seen_keys:
      return
    seen_keys.add(credential.key)
    credentials.append(credential)

  def _append_token(token: str) -> None:
    stored = by_access_token.get(token)
    if stored is None:
      _append(AnthropicCredential(key=token, token=token))
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
  for name, read_form in _ENV_POOL_FORMS:
    for token in read_form(env.get(name, "")):
      _append_token(token)
  _append_records(*stores[-1])

  return AnthropicCredentialSources(
    credentials=tuple(credentials),
    store_path=org_path,
    user_scope=scope,
    user_store_path=user_path,
  )


class AnthropicCredentialPool:
  """Which sibling credential is usable right now, for this process.

  The resolver owns the order; the limiter's own reset headers own the block
  windows. Selection is a read of those two and never a decision of its own.
  Keys are credential keys (an enrolled account's identity, or an environment
  token), so a refreshed access token stays parked for its account's window.
  """

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


# Usage limits outlive a single request and a single session, so the block state
# is process-wide: a credential the limiter rejected for the next five hours must
# not be handed to the next session that starts.
ANTHROPIC_CREDENTIAL_POOL = AnthropicCredentialPool()
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
  except httpx.HTTPError as exc:
    raise AnthropicOAuthError(f"Anthropic OAuth token request failed: {exc}") from exc
  if response.status_code >= 400:
    detail = str(response.text or "").strip()[:200]
    raise AnthropicOAuthError(
      f"Anthropic OAuth token request failed ({response.status_code}): {detail}"
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
  except httpx.HTTPError as exc:
    raise AnthropicOAuthError(f"Anthropic OAuth profile request failed: {exc}") from exc
  if response.status_code >= 400:
    detail = str(response.text or "").strip()[:200]
    raise AnthropicOAuthError(
      f"Anthropic OAuth profile request failed ({response.status_code}): {detail}"
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
  """Exchange this account's refresh token, persisting the rotated one.

  Anthropic rotates the refresh token on use, so the new one is written before
  this function returns: an interrupted caller must not leave the store holding
  a grant the server has already retired. A failed refresh is recorded on the
  account and re-raised; the record is never deleted, because the operator's
  remedy is one `auth login anthropic` for that account, not a vanished row.
  """

  if not record.refresh_token:
    raise AnthropicOAuthError(
      f"Anthropic account {record.identity} has no refresh token; run "
      "`python3 -m agent_gateway.cli auth login anthropic` for it"
    )
  moment = time.time() if now is None else now
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
  except AnthropicOAuthError as exc:
    failed = replace(record, last_error=str(exc)[:300], last_error_at=moment)
    upsert_anthropic_oauth_record(store_path, failed)
    raise
  refreshed = replace(
    record,
    access_token=access_token,
    # An OAuth server that omits refresh_token leaves the presented one valid;
    # Anthropic sends a new one, and that is the one that must land on disk.
    refresh_token=refresh_token or record.refresh_token,
    expires_at=moment + expires_in if expires_in else moment,
    updated_at=moment,
    scope=scope or record.scope,
    last_error="",
    last_error_at=0.0,
  )
  upsert_anthropic_oauth_record(store_path, refreshed)
  LOGGER.info(
    "Refreshed Anthropic OAuth access token for %s (expires in %.0fs)",
    refreshed.identity,
    max(0.0, refreshed.expires_at - moment),
  )
  return refreshed


def _usable_token(
  sources: AnthropicCredentialSources,
  candidates: Sequence[AnthropicCredential],
  *,
  pool: AnthropicCredentialPool,
  now: float | None = None,
) -> str:
  """The first candidate that is neither parked nor stale, refreshing as needed."""

  moment = time.time() if now is None else now
  for credential in candidates:
    if pool.blocked_until(credential.key) > moment:
      continue
    record = credential.record
    store_path = credential.store_path
    if record is None or store_path is None or not record.needs_refresh(now=moment):
      return credential.token
    try:
      refreshed = refresh_anthropic_oauth_record(
        record,
        store_path=store_path,
        now=moment,
      )
    except AnthropicOAuthError as exc:
      # Same mechanism as a usage limit: park this account and let the next
      # sibling serve. The next attempt retries the refresh.
      pool.block(credential.key, until=moment + REFRESH_FAILURE_BLOCK_SECONDS)
      LOGGER.warning(
        "Anthropic OAuth refresh failed for %s; parking it for %.0fs: %s",
        credential.key,
        REFRESH_FAILURE_BLOCK_SECONDS,
        exc,
      )
      continue
    return refreshed.access_token
  return ""


def _log_pool_summary(sources: AnthropicCredentialSources) -> None:
  summary = ", ".join(
    f"{credential.key if credential.record is not None else 'env'}"
    f"={mask_anthropic_token(credential.token)}"
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


def select_anthropic_credential(
  sources: AnthropicCredentialSources,
  *,
  pool: AnthropicCredentialPool | None = None,
  now: float | None = None,
) -> str:
  """Bind a credential: the first usable one, else the first at all.

  A pool of one is never withheld from a run — an expired or unrefreshable
  single credential still binds, so the turn walks into the provider's existing
  rate-limit/auth handling instead of failing here with no request made.
  """

  selected_pool = ANTHROPIC_CREDENTIAL_POOL if pool is None else pool
  _log_pool_summary(sources)
  return _usable_token(sources, sources.credentials, pool=selected_pool, now=now) or (
    sources.credentials[0].token if sources.credentials else ""
  )


def rotate_anthropic_credential(
  sources: AnthropicCredentialSources,
  *,
  current: str,
  until: float,
  pool: AnthropicCredentialPool | None = None,
  now: float | None = None,
) -> str:
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
  return _usable_token(sources, siblings, pool=selected_pool, now=now)


def ensure_fresh_anthropic_credential(
  token: str,
  *,
  config: Mapping[str, Any] | None = None,
  environ: Mapping[str, str] | None = None,
  now: float | None = None,
) -> str:
  """The access token to send for `token`, refreshed first when it is stale.

  A credential supplied through the environment has no refresh token and is
  returned untouched, with no store read beyond the one the resolver already
  does. An enrolled account whose access token expired refreshes here, at the
  boundary where the credential is about to be bound to a client.
  """

  normalized = str(token or "").strip()
  if not normalized:
    return ""
  sources = resolve_anthropic_credentials(config, environ=environ)
  credential = sources.by_token(normalized)
  if credential is None or credential.record is None or credential.store_path is None:
    return normalized
  record = credential.record
  if not record.needs_refresh(now=now):
    return normalized
  try:
    return refresh_anthropic_oauth_record(
      record,
      store_path=credential.store_path,
      now=now,
    ).access_token
  except AnthropicOAuthError as exc:
    # The bound credential is still handed to the request: the provider's 401
    # is what rotates the run to a sibling, and withholding it here would end
    # the turn with no request made at all.
    LOGGER.warning("Anthropic OAuth refresh failed for %s: %s", record.identity, exc)
    return normalized


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
  upsert_anthropic_oauth_record(store_path, record)
  return record


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
  upsert_anthropic_oauth_record(store_path, record)
  return record



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
  "REFRESH_FAILURE_BLOCK_SECONDS",
  "REFRESH_MARGIN_SECONDS",
  "UNDECLARED_LIMIT_BLOCK_SECONDS",
  "anthropic_token_store_is_private",
  "anthropic_user_scope",
  "complete_anthropic_login",
  "enroll_anthropic_setup_token",
  "ensure_fresh_anthropic_credential",
  "generate_pkce",
  "load_anthropic_oauth_store",
  "mask_anthropic_token",
  "parse_anthropic_callback",
  "refresh_anthropic_oauth_record",
  "remove_anthropic_oauth_record",
  "resolve_anthropic_auth_store_path",
  "resolve_anthropic_credentials",
  "rotate_anthropic_credential",
  "save_anthropic_oauth_store",
  "select_anthropic_credential",
  "setup_token_account_identity",
  "start_anthropic_login",
  "upsert_anthropic_oauth_record",
]
