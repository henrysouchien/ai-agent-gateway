from __future__ import annotations

import asyncio
import fcntl
import functools
import json
import os
from pathlib import Path
import signal
import time
from typing import Any, Mapping

from .autonomous_capability_handoff import (
  AutonomousCapabilityBindingRequest,
  resolve_autonomous_capability_binding,
)
from .autonomous_admission_ledger import (
  prepare_autonomous_admission_ledger,
)
from .autonomous_control_files import (
  fsync_owned_file_directory,
  secure_create_owned_file,
  unlink_created_owned_file,
)
from .autonomous_approval_channel import (
  AUTONOMOUS_APPROVAL_CHANNEL_FD_ENV,
  AutonomousApprovalChannelAuthority,
  AutonomousApprovalChannelChild,
  create_autonomous_approval_channel,
)
from .autonomous_credential_handoff import (
  AUTONOMOUS_CREDENTIAL_HANDOFF_ENV,
  AUTONOMOUS_CREDENTIAL_HANDOFF_MAX_BYTES,
  AUTONOMOUS_CREDENTIAL_HANDOFF_STDIN,
  encode_autonomous_credential_handoff,
)
from .autonomous_launch_envelope import (
  AUTONOMOUS_CAPABILITY_ENVELOPE_ENV,
  AUTONOMOUS_RUNTIME_SESSION_PURPOSE,
  AutonomousControlAuthority,
  AutonomousDispatchScope,
  AutonomousLaunchWorkload,
  AutonomousSessionAuthority,
  OrdinaryAutonomousSessionAuthority,
)
from .autonomous_claim_broker import (
  AUTONOMOUS_CLAIM_BROKER_FD_ENV,
  AutonomousClaimBroker,
)
from .autonomous_event_channel import (
  AUTONOMOUS_EVENT_CHANNEL_FD_ENV,
  AUTONOMOUS_EVENT_CHANNEL_MAX_IO_TIMEOUT_SECONDS,
  AutonomousEventChannelChild,
  create_autonomous_event_channel,
)
from .claim_signing_authority import GatewayClaimSigningAuthority
from .autonomous_runner_commands import normalize_autonomous_profile, normalize_max_budget_usd
from .artifact_paths import canonicalize_ticker
from .capability_binding import CredentialHandle
from .events import DEFAULT_SCHEMA_VERSION
from .role_validation import require_exact_role
from .autonomous_runner_state import (
  AutonomousTask,
  _fallback_identity_payload,
  _normalize_dispatch_scope,
  _normalize_identity_aliases,
  _positive_int,
  _runtime_attr,
  _user_identity_api,
  autonomous_owner_lease_is_released,
)

_SPAWN_CLEANUP_GRACE_SEC = 1.0
_AUTONOMOUS_SESSION_AUTHORITY_TTL_SECONDS = 24 * 60 * 60
_AUTONOMOUS_CHILD_BASE_ENV_NAMES = frozenset({
  "PATH",
  "PYTHONPATH",
  "PYTHONHOME",
  "VIRTUAL_ENV",
  "HOME",
  "USER",
  "LOGNAME",
  "SHELL",
  "LANG",
  "LC_ALL",
  "LC_CTYPE",
  "TZ",
  "TMPDIR",
  "SSL_CERT_FILE",
  "SSL_CERT_DIR",
  "REQUESTS_CA_BUNDLE",
  "NO_PROXY",
  "PRODUCT_ID",
  "ENVIRONMENT",
  "APP_ENV",
  "AGENT_GATEWAY_ENV",
  "AGENT_GATEWAY_RATES_FILE",
  "USER_DATA_DIR",
  "CORPUS_LOG_DIR",
  "CORPUS_STATE_DIR",
  "LOG_DIR",
  "GATEWAY_LOG_DIR",
  "MCP_CONFIG_TMP_DIR",
  "MCP_CONFIG_TEMPLATE",
  "MCP_STARTUP_CONCURRENCY",
  "MCP_STDIO_CONNECT_BACKOFF_S",
  "MCP_STDIO_CONNECT_RETRIES",
  "MCP_STDIO_CONNECT_STABILIZE_S",
  "LOCAL_GATEWAY_PYTHON",
  "LOCAL_GATEWAY_CONTROL_GENERATION",
  "LOCAL_GATEWAY_CONTROL_GENERATION_ID",
  "LOCAL_GATEWAY_RUNTIME_ROOT",
  "LOCAL_GATEWAY_RUNTIME_MANIFEST",
  "LOCAL_GATEWAY_RUNTIME_VERSION_ROOT",
  "GATEWAY_URL",
  "GATEWAY_BASE_URL",
  "GATEWAY_SSL_VERIFY",
  "AGENT_API_CLAIM_TTL_SECONDS",
  "AGENT_API_CLAIM_MAX_TTL_SECONDS",
  "AGENT_GATEWAY_SKILLS_DIR",
  "AGENT_SESSION_LOG_BASE_DIR",
  "AGENT_SESSION_LOG_MAX_ACTIVE_BYTES",
  "AGENT_SDK_CWD",
  "SUB_AGENT_MAX_CONCURRENCY",
  "SUB_AGENT_SKILL_TIMEOUT",
  "SUB_AGENT_TIMEOUT",
  "PARENT_PER_TURN_TIMEOUT",
  "MEMORY_ENABLED",
  "MEMORY_KEYWORD_WEIGHT",
  "MEMORY_VECTOR_WEIGHT",
  "MEMORY_MAX_PROMPT_TOKENS",
  "CITATION_SESSION_REGISTRY_RETENTION_DAYS",
  "EDGAR_USE_UPSTREAM_EQUIVALENCE",
  "RISK_API_URL",
  "RISK_CLIENT_PATH",
  "RISK_MODULE_RESOLVER_URL",
  "BROKERAGE_CONNECT_REQUIRED",
  "CODE_EXECUTE_DOCKER_IMAGE",
  "EDGAR_API_URL",
  "EDGAR_UPDATER_ROOT",
  "EXCEL_MCP_BACKEND_URL",
  "MODEL_ENGINE_OVERRIDES_DIR",
})
_AUTONOMOUS_CHILD_PROVIDER_ENV_NAMES = {
  "anthropic": frozenset(),
  "openai": frozenset({
    "OPENAI_SESSION_EPOCH",
  }),
  "codex": frozenset(),
  "xai": frozenset(),
}
_AUTONOMOUS_CHILD_RESEARCH_TOOL_ENV_NAMES = frozenset({
  "FMP_API_KEY",
  "FMP_CACHE_DIR",
  "FRED_API_KEY",
  "ORATS_API_KEY",
  "EDGAR_API_KEY",
  "SEC_USER_AGENT",
  "DATABENTO_API_KEY",
  "ALERTS_GATEWAY_API_KEY",
})
_AUTONOMOUS_CHILD_PROFILE_TOOL_ENV_NAMES = {
  "analyst": _AUTONOMOUS_CHILD_RESEARCH_TOOL_ENV_NAMES,
  "research-producer": _AUTONOMOUS_CHILD_RESEARCH_TOOL_ENV_NAMES,
  "advisor": (
    _AUTONOMOUS_CHILD_RESEARCH_TOOL_ENV_NAMES
    | frozenset({
      "IBKR_ENABLED",
      "IBKR_FLEX_ENABLED",
      "IBKR_FLEX_TOKEN",
      "IBKR_FLEX_QUERY_ID",
      "SCHWAB_CLIENT_ID",
      "SCHWAB_CLIENT_SECRET",
      "SNAPTRADE_CLIENT_ID",
      "SNAPTRADE_CONSUMER_KEY",
      "PLAID_CLIENT_ID",
      "PLAID_SECRET",
      "PLAID_ENV",
    })
  ),
}
_AUTONOMOUS_CHILD_PROFILE_ENV_NAMES = {
  profile: frozenset({
    f"{prefix}_AGENT_CLIENT_TIMEOUT",
    f"{prefix}_AGENT_MAX_BUDGET_USD",
    f"{prefix}_AGENT_MAX_TOKENS",
    f"{prefix}_AGENT_MAX_TURNS",
    f"{prefix}_AGENT_STREAM_STALL_TIMEOUT",
    f"{prefix}_AGENT_TIMEOUT_SECONDS",
  })
  for profile, prefix in (
    ("analyst", "ANALYST"),
    ("advisor", "ADVISOR"),
    ("research-producer", "RESEARCH_PRODUCER"),
  )
}
_RETIRED_AUTONOMOUS_CHILD_ENV_NAMES = (
  "AUTONOMOUS_USER_ID",
  "AUTONOMOUS_RAW_USER_ID",
  "AUTONOMOUS_USER_SLUG",
  "AUTONOMOUS_USER_EMAIL",
  "AGENT_AUTONOMOUS_EVENTS_PATH",
  "AGENT_AUTONOMOUS_TASK_ID",
  "AGENT_AUTONOMOUS_CONTROL_RUN_ID",
  "AGENT_AUTONOMOUS_CONTROL_CHANNEL",
  "AGENT_AUTONOMOUS_GATEWAY_SESSION_ID",
  "AGENT_AUTONOMOUS_DISPATCH_SCOPE_JSON",
  "AGENT_AUTONOMOUS_OPERATOR_INBOX_PATH",
  "AGENT_AUTONOMOUS_APPROVAL_DECISIONS_PATH",
  "AGENT_AUTONOMOUS_APPROVALS_DB_PATH",
  "AGENT_API_USER_CLAIM_HMAC_KEY",
  "AGENT_API_CLAIM_AUDIENCE",
  "AGENT_API_CLAIM_ISSUED_AT",
  "AGENT_API_CLAIM_EXPIRY",
  "AGENT_API_CLAIM_USER_ID",
  "AGENT_API_CLAIM_USER_EMAIL",
  "AGENT_API_CLAIM_NONCE",
  "AGENT_API_CLAIM_SIGNATURE",
  AUTONOMOUS_CLAIM_BROKER_FD_ENV,
  AUTONOMOUS_APPROVAL_CHANNEL_FD_ENV,
  AUTONOMOUS_CREDENTIAL_HANDOFF_ENV,
  "AGENT_AUTONOMOUS_USER_CREDENTIAL_HANDOFF",
  "ANALYST_DEV_MODE",
  "ADVISOR_DEV_MODE",
  "RESEARCH_PRODUCER_DEV_MODE",
)


def _pinned_autonomous_child_pythonpath(
  environ: Mapping[str, str],
) -> str | None:
  """Return the verified paired-runtime import roots for a spawned child."""

  raw_version_root = str(
    environ.get("LOCAL_GATEWAY_RUNTIME_VERSION_ROOT") or ""
  ).strip()
  if not raw_version_root:
    return None
  version_root = Path(raw_version_root)
  if not version_root.is_absolute():
    raise ValueError(
      "LOCAL_GATEWAY_RUNTIME_VERSION_ROOT must be absolute"
    )
  ai_root = version_root / "ai-excel-addin"
  risk_root = version_root / "risk_module"
  return os.pathsep.join(str(path) for path in (
    ai_root,
    ai_root / "api",
    ai_root / "packages" / "agent-gateway",
    ai_root / "packages" / "excel-mcp" / "python",
    ai_root / "packages" / "ibkr-relay-client" / "python",
    ai_root / "packages" / "sheets-finance-mcp",
    ai_root / "packages" / "value-semantics-core",
    risk_root / "brokerage-connect",
    risk_root,
  ))


def _positive_autonomous_child_env(
  environ: Mapping[str, str],
  *,
  provider: str,
  profile: str,
  deliver: bool,
) -> dict[str, str]:
  """Project gateway state onto the child's explicit runtime contract."""

  normalized_provider = str(provider or "").strip().lower()
  normalized_profile = str(profile or "").strip().lower()
  if type(deliver) is not bool:
    raise TypeError(
      "autonomous child environment deliver flag must be exact bool"
    )
  allowed_names = (
    _AUTONOMOUS_CHILD_BASE_ENV_NAMES
    | _AUTONOMOUS_CHILD_PROVIDER_ENV_NAMES.get(
      normalized_provider,
      frozenset(),
    )
    | _AUTONOMOUS_CHILD_PROFILE_ENV_NAMES.get(
      normalized_profile,
      frozenset(),
    )
    | _AUTONOMOUS_CHILD_PROFILE_TOOL_ENV_NAMES.get(
      normalized_profile,
      frozenset(),
    )
  )
  projected = {
    name: value
    for name in allowed_names
    if type((value := environ.get(name))) is str
  }
  if not projected.get("PYTHONPATH"):
    pinned_pythonpath = _pinned_autonomous_child_pythonpath(environ)
    if pinned_pythonpath is not None:
      projected["PYTHONPATH"] = pinned_pythonpath
  if (
    deliver
    and normalized_profile
    in _AUTONOMOUS_CHILD_PROFILE_ENV_NAMES
  ):
    profile_prefix = normalized_profile.upper().replace("-", "_")
    for suffix in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
      destination = f"{profile_prefix}_{suffix}"
      value = (
        environ.get(destination)
        or environ.get(suffix)
      )
      if type(value) is str and value:
        projected[destination] = value
  return projected


_OWNED_PROCESS_SENTINEL_SOURCE = r"""
import json
import fcntl
import os
import select
import signal
import subprocess
import sys
import threading
import time

event_fd = int(os.environ["AGENT_AUTONOMOUS_EVENT_CHANNEL_FD"])
claim_broker_fd = int(
    os.environ["AGENT_AUTONOMOUS_CLAIM_BROKER_FD"]
)
approval_fd_raw = os.environ.get(
    "AGENT_AUTONOMOUS_APPROVAL_CHANNEL_FD"
)
approval_fd = int(approval_fd_raw) if approval_fd_raw is not None else None
child_descriptors = (
    (event_fd, claim_broker_fd, approval_fd)
    if approval_fd is not None
    else (event_fd, claim_broker_fd)
)
credential_max_bytes = int(sys.argv[1])
owner_lifeline_fd = int(sys.argv[2])
owner_lease_fd = int(sys.argv[3])
owner_cleanup_grace_seconds = float(sys.argv[4])
target_cmd = sys.argv[5:]


def emit_status(payload):
    sys.stderr.write(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    sys.stderr.flush()


def park():
    while True:
        signal.pause()


def close_child_descriptors():
    for fd in child_descriptors:
        try:
            os.close(fd)
        except OSError:
            pass


def ignore_shutdown_signal(_signal_number, _frame):
    return None


signal.signal(signal.SIGTERM, ignore_shutdown_signal)
signal.signal(signal.SIGINT, ignore_shutdown_signal)


try:
    fcntl.flock(owner_lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BaseException as exc:
    close_child_descriptors()
    os.close(owner_lease_fd)
    emit_status({
        "error": "owner_lease_failed:" + type(exc).__name__,
        "kind": "error",
        "version": 1,
    })
    try:
        while os.read(owner_lifeline_fd, 1):
            pass
    finally:
        raise SystemExit(125)


def monitor_owner():
    try:
        while os.read(owner_lifeline_fd, 1):
            pass
    except BaseException:
        pass
    finally:
        try:
            os.close(owner_lifeline_fd)
        except OSError:
            pass
    terminate_owned_group()


def terminate_owned_group():
    process_group_id = os.getpgrp()
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + owner_cleanup_grace_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.05, remaining))
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass


def owner_lifeline_closed():
    ready, _, _ = select.select([owner_lifeline_fd], [], [], 0)
    if not ready:
        return False
    try:
        return os.read(owner_lifeline_fd, 1) == b""
    except OSError:
        return True

if (
    os.environ.get("AGENT_AUTONOMOUS_CREDENTIAL_HANDOFF")
    != "stdin-json-v1"
):
    close_child_descriptors()
    emit_status({
        "error": "credential_handoff_missing",
        "kind": "error",
        "version": 1,
    })
    try:
        while os.read(owner_lifeline_fd, 1):
            pass
    finally:
        raise SystemExit(125)
credential_payload = sys.stdin.buffer.read(credential_max_bytes + 1)
if not credential_payload or len(credential_payload) > credential_max_bytes:
    close_child_descriptors()
    emit_status({
        "error": "credential_handoff_invalid_size",
        "kind": "error",
        "version": 1,
    })
    try:
        while os.read(owner_lifeline_fd, 1):
            pass
    finally:
        raise SystemExit(125)

try:
    from agent_gateway.secret_boundary import (
        SANITIZATION_FAILED,
        SecretBoundary,
    )

    handoff_payload = json.loads(credential_payload.decode("utf-8"))
    auth_config = handoff_payload.get("auth_config")
    output_boundary = SecretBoundary.from_auth_config(auth_config)
except BaseException as exc:
    close_child_descriptors()
    emit_status({
        "error": "credential_boundary_registration_failed:" + type(exc).__name__,
        "kind": "error",
        "version": 1,
    })
    try:
        while os.read(owner_lifeline_fd, 1):
            pass
    finally:
        raise SystemExit(125)

if owner_lifeline_closed():
    close_child_descriptors()
    terminate_owned_group()
    park()

try:
    child = subprocess.Popen(
        target_cmd,
        close_fds=True,
        pass_fds=child_descriptors,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
except BaseException as exc:
    threading.Thread(
        target=monitor_owner,
        name="autonomous-owner-lifeline",
        daemon=True,
    ).start()
    close_child_descriptors()
    emit_status({
        "error": "child_spawn_failed:" + type(exc).__name__,
        "kind": "error",
        "version": 1,
    })
    park()

threading.Thread(
    target=monitor_owner,
    name="autonomous-owner-lifeline",
    daemon=True,
).start()
close_child_descriptors()
projection_failed = threading.Event()


def project_child_output():
    source = child.stdout
    if source is None:
        projection_failed.set()
        return
    try:
        while True:
            raw_line = source.readline(credential_max_bytes + 1)
            if not raw_line:
                break
            if (
                len(raw_line) > credential_max_bytes
                and not raw_line.endswith(b"\n")
            ):
                projection_failed.set()
                sys.stdout.buffer.write((SANITIZATION_FAILED + "\n").encode("utf-8"))
                sys.stdout.buffer.flush()
                while source.read(credential_max_bytes):
                    pass
                return
            try:
                projected = output_boundary.sanitize(
                    raw_line.decode("utf-8", errors="replace"),
                    sink="autonomous_child_output_log",
                )
                if type(projected) is not str:
                    raise TypeError("child output projection must be text")
            except BaseException:
                projected = SANITIZATION_FAILED
                projection_failed.set()
            sys.stdout.buffer.write(projected.encode("utf-8"))
            sys.stdout.buffer.flush()
    except BaseException:
        projection_failed.set()
        try:
            sys.stdout.buffer.write((SANITIZATION_FAILED + "\n").encode("utf-8"))
            sys.stdout.buffer.flush()
        except BaseException:
            pass
    finally:
        try:
            source.close()
        except BaseException:
            pass


output_thread = threading.Thread(
    target=project_child_output,
    name="autonomous-child-output-projection",
    daemon=True,
)
output_thread.start()
try:
    child.stdin.write(credential_payload)
    child.stdin.close()
except BaseException as exc:
    emit_status({
        "error": "credential_handoff_failed:" + type(exc).__name__,
        "kind": "error",
        "version": 1,
    })
    park()

returncode = child.wait()
output_thread.join()
if projection_failed.is_set():
    emit_status({
        "error": "child_output_projection_failed",
        "kind": "error",
        "version": 1,
    })
    park()
emit_status({
    "kind": "exited",
    "returncode": returncode,
    "version": 1,
})
park()
"""


def _with_run_mutation_lock(func):
  @functools.wraps(func)
  async def guarded(self, *args, **kwargs):
    async with self.run_mutation_lock:
      return await func(self, *args, **kwargs)

  return guarded


def _asyncio_module() -> Any:
  return _runtime_attr("asyncio", asyncio)


def _get_process_group_id(pid: int) -> int:
  return os.getpgid(pid)


def _signal_process_group(process_group_id: int, signal_number: int) -> None:
  os.killpg(process_group_id, signal_number)


def _create_owner_lifeline() -> tuple[int, int]:
  pipe2 = getattr(os, "pipe2", None)
  if callable(pipe2):
    read_fd, write_fd = pipe2(getattr(os, "O_CLOEXEC", 0))
  else:
    read_fd, write_fd = os.pipe()
    os.set_inheritable(read_fd, False)
    os.set_inheritable(write_fd, False)
  if os.get_inheritable(read_fd) or os.get_inheritable(write_fd):
    os.close(read_fd)
    os.close(write_fd)
    raise RuntimeError(
      "autonomous owner lifeline must be close-on-exec"
    )
  return read_fd, write_fd


def _narrowed_mcp_gateway_user_keys(
  *,
  api_dir: Path,
  user_id: str,
  user_email: str | None,
) -> str:
  try:
    api = _user_identity_api(api_dir=api_dir)
  except (Exception, SystemExit) as exc:
    raise RuntimeError(
      "autonomous spawn refused: user identity API import failed"
    ) from exc
  lookup = getattr(api, "get_mcp_user_key_entry", None) if api is not None else None
  if not callable(lookup):
    raise RuntimeError(
      "autonomous spawn refused: user identity API is unavailable"
    )
  try:
    entry = lookup(user_id, user_email)
  except SystemExit as exc:
    raise RuntimeError(
      "autonomous spawn refused: GATEWAY_USER_KEYS is malformed"
    ) from exc
  if entry is None:
    raise RuntimeError(
      "autonomous spawn refused: no GATEWAY_USER_KEYS "
      f"channel='mcp' entry for user {user_id!r}"
    )
  if type(entry) is not dict:
    raise RuntimeError(
      "autonomous spawn refused: GATEWAY_USER_KEYS entry is malformed"
    )
  key = entry.get("key")
  channel = entry.get("channel")
  if type(channel) is not str or channel != "mcp":
    raise RuntimeError(
      "autonomous spawn refused: GATEWAY_USER_KEYS entry channel is invalid"
    )
  if type(key) is not str or not key.strip():
    raise RuntimeError(
      "autonomous spawn refused: GATEWAY_USER_KEYS entry key is empty"
    )
  narrowed_entry = {
    field: entry.get(field)
    for field in ("key", "slug", "email", "risk_user_id", "channel", "role")
  }
  try:
    return json.dumps([narrowed_entry])
  except (TypeError, ValueError) as exc:
    raise RuntimeError(
      "autonomous spawn refused: GATEWAY_USER_KEYS entry is malformed"
    ) from exc


def _start_identity_payload(
  *,
  raw_user_id: str,
  user_email: str | None,
  owner_user_id: str | None,
  user_slug: str | None,
  risk_user_id: int | None,
  user_aliases: list[str] | tuple[str, ...] | None,
  identity_status: str | None,
) -> dict[str, Any]:
  explicit_owner = str(owner_user_id or "").strip() or None
  normalized_slug = str(user_slug or "").strip() or None
  normalized_risk = _positive_int(risk_user_id)

  if explicit_owner is None:
    try:
      api = _user_identity_api()
    except (Exception, SystemExit) as exc:
      raise RuntimeError(
        "autonomous spawn refused: user identity API import failed"
      ) from exc
    resolver = getattr(api, "resolve_canonical_user_identity", None) if api is not None else None
    if callable(resolver):
      try:
        identity = resolver(
          raw_user_id,
          risk_user_id=risk_user_id,
          user_email=user_email,
          mapped_slug=normalized_slug,
          allow_legacy_fallback=True,
        )
        return {
          "owner_user_id": str(identity.owner_user_id),
          "raw_user_id": raw_user_id,
          "user_slug": identity.user_slug,
          "risk_user_id": int(identity.risk_user_id),
          "user_aliases": _normalize_identity_aliases(
            identity.owner_user_id,
            identity.raw_user_id,
            identity.user_slug,
            identity.user_email,
            identity.aliases,
            user_aliases,
          ),
          "identity_status": identity_status or str(identity.identity_status),
        }
      except SystemExit as exc:
        raise RuntimeError(
          "autonomous spawn refused: GATEWAY_USER_KEYS is malformed"
        ) from exc
      except ValueError as exc:
        raise ValueError(f"Unable to resolve canonical autonomous user identity for {raw_user_id!r}") from exc

  fallback_owner = explicit_owner or (str(normalized_risk) if normalized_risk is not None else raw_user_id)
  fallback_slug = normalized_slug
  if fallback_slug is None and raw_user_id != fallback_owner and not raw_user_id.isdecimal():
    fallback_slug = raw_user_id
  return _fallback_identity_payload(
    user_id=fallback_owner,
    user_email=user_email,
    identity_status=identity_status
    or ("risk_user_id_authoritative" if normalized_risk is not None else "legacy_user_id_fallback"),
    risk_user_id=normalized_risk,
    owner_user_id=fallback_owner,
    raw_user_id=raw_user_id,
    user_slug=fallback_slug,
    user_aliases=list(user_aliases or ()),
  )


class AutonomousRegistryStartMixin:
  def _owned_process_sentinel_cmd(
    self,
    target_cmd: list[str],
    *,
    owner_lifeline_fd: int,
    owner_lease_fd: int,
  ) -> list[str]:
    return [
      self._python,
      "-c",
      _OWNED_PROCESS_SENTINEL_SOURCE,
      str(AUTONOMOUS_CREDENTIAL_HANDOFF_MAX_BYTES),
      str(owner_lifeline_fd),
      str(owner_lease_fd),
      str(_SPAWN_CLEANUP_GRACE_SEC),
      *target_cmd,
    ]

  def _close_owner_lifeline(
    self,
    record: AutonomousTask | None,
  ) -> None:
    if record is None:
      return
    owner_lifeline_fd = record.owner_lifeline_fd
    if owner_lifeline_fd is None:
      return
    record.owner_lifeline_fd = None
    os.close(owner_lifeline_fd)

  def _require_released_owner_lease(
    self,
    record: AutonomousTask,
  ) -> None:
    if not autonomous_owner_lease_is_released(
      record.owner_lease_path,
      expected_device=record.owner_lease_device,
      expected_inode=record.owner_lease_inode,
    ):
      raise RuntimeError(
        "prior autonomous run owner cleanup is still active"
      )

  async def _reserve_slot(self) -> None:
    async with self._slot_lock:
      if self._reserved_slots >= self._max_running:
        raise RuntimeError(f"Autonomous concurrency limit reached ({self._max_running})")
      self._reserved_slots += 1

  async def _release_slot(self, record: AutonomousTask | None = None) -> None:
    async with self._slot_lock:
      if record is None:
        self._reserved_slots = max(0, self._reserved_slots - 1)
        return
      if not record.slot_reserved:
        return
      record.slot_reserved = False
      self._reserved_slots = max(0, self._reserved_slots - 1)

  async def _await_cleanup(self, cleanup_coro) -> None:
    asyncio_module = _asyncio_module()
    cleanup_task = asyncio_module.create_task(cleanup_coro)
    try:
      await asyncio_module.shield(cleanup_task)
    except asyncio_module.CancelledError:
      await cleanup_task
      raise

  def _claim_owned_process_group(self, record: AutonomousTask) -> None:
    proc = record.proc
    pid = getattr(proc, "pid", None)
    if type(pid) is not int or pid <= 0:
      raise RuntimeError(
        "autonomous child did not expose a valid process identity"
      )
    try:
      process_group_id = _runtime_attr(
        "_get_process_group_id",
        _get_process_group_id,
      )(pid)
    except ProcessLookupError as exc:
      raise RuntimeError(
        "autonomous child exited before process-group ownership was established"
      ) from exc
    if type(process_group_id) is not int or process_group_id != pid:
      raise RuntimeError(
        "autonomous child is not the leader of its requested private process group"
      )
    record.process_group_pid = pid
    record.process_group_id = process_group_id

  def _signal_owned_process_group(
    self,
    record: AutonomousTask,
    signal_number: int,
  ) -> bool:
    proc = record.proc
    pid = record.process_group_pid
    process_group_id = record.process_group_id
    if (
      proc is None
      or type(pid) is not int
      or type(process_group_id) is not int
      or pid <= 0
      or process_group_id != pid
      or getattr(proc, "pid", None) != pid
    ):
      return False
    if proc.returncode is None:
      try:
        live_process_group_id = _runtime_attr(
          "_get_process_group_id",
          _get_process_group_id,
        )(pid)
      except ProcessLookupError:
        return False
      except OSError as exc:
        raise RuntimeError(
          f"autonomous process-group lookup failed: {exc}"
        ) from exc
      if live_process_group_id != process_group_id:
        raise RuntimeError(
          "autonomous process-group identity changed before signalling"
        )
    else:
      return False
    try:
      _runtime_attr(
        "_signal_process_group",
        _signal_process_group,
      )(process_group_id, signal_number)
    except ProcessLookupError:
      return False
    except OSError as exc:
      # Darwin can fail kill/killpg with EPERM (PermissionError) while a
      # group member is mid-exit. Surface OS-level signal failures through
      # the RuntimeError contract that every reaper/cancel call site already
      # treats as a recorded cleanup failure, instead of letting an OSError
      # escape and kill the reaper task with the run stuck in "running".
      raise RuntimeError(
        f"autonomous process-group signal failed: {exc}"
      ) from exc
    return True

  async def _terminate_owned_process(self, record: AutonomousTask | None) -> None:
    self._close_owner_lifeline(record)
    proc = None if record is None else record.proc
    if proc is None:
      return
    if proc.returncode is None:
      if record is not None and record.process_group_id is not None:
        self._signal_owned_process_group(record, signal.SIGTERM)
      else:
        try:
          proc.terminate()
        except ProcessLookupError:
          return
      asyncio_module = _asyncio_module()
      try:
        await asyncio_module.wait_for(
          proc.wait(),
          timeout=_runtime_attr("_SPAWN_CLEANUP_GRACE_SEC", _SPAWN_CLEANUP_GRACE_SEC),
        )
      except asyncio_module.TimeoutError:
        if record is not None and record.process_group_id is not None:
          self._signal_owned_process_group(record, signal.SIGKILL)
        else:
          try:
            proc.kill()
          except ProcessLookupError:
            pass
        try:
          await asyncio_module.wait_for(
            proc.wait(),
            timeout=_runtime_attr(
              "_SPAWN_CLEANUP_GRACE_SEC",
              _SPAWN_CLEANUP_GRACE_SEC,
            ),
          )
        except asyncio_module.TimeoutError as exc:
          raise RuntimeError(
            "autonomous process did not exit after final cleanup signal"
          ) from exc

  async def _spawn_owned_process(
    self,
    record: AutonomousTask,
    spawn_coro,
  ) -> None:
    asyncio_module = _asyncio_module()
    spawn_task = asyncio_module.create_task(spawn_coro)
    try:
      record.proc = await asyncio_module.shield(spawn_task)
    except asyncio_module.CancelledError:
      record.proc = await spawn_task
      self._claim_owned_process_group(record)
      raise
    self._claim_owned_process_group(record)

  async def _cleanup_uncommitted_start(
    self,
    *,
    task_id: str,
    record: AutonomousTask | None,
    log_handle: Any | None,
    event_channel_child: AutonomousEventChannelChild | None,
    approval_channel_child: AutonomousApprovalChannelChild | None,
  ) -> None:
    self._tasks.pop(task_id, None)
    if event_channel_child is not None:
      event_channel_child.close()
    if approval_channel_child is not None:
      approval_channel_child.close()
    try:
      await self._terminate_owned_process(record)
    finally:
      if record is not None and record.claim_broker is not None:
        record.claim_broker.close()
        record.claim_broker = None
      if record is not None and record.event_channel is not None:
        record.event_channel.close()
      if record is not None and record.approval_channel is not None:
        record.approval_channel.close()
        record.approval_channel = None
      if record is not None and record.event_channel_task is not None:
        await _asyncio_module().gather(
          record.event_channel_task,
          return_exceptions=True,
        )
      if log_handle is not None:
        log_handle.close()
      if record is not None:
        record.log_handle = None
        await self._release_slot(record)
      else:
        await self._release_slot()
      if record is not None and record.tool_result_spill_dir is not None:
        self._remove_registered_tool_result_spill_dir(
          task_id,
          record.tool_result_spill_dir,
        )
      self._delete_task_manifest(task_id)

  def _signed_launch_envelope(
    self,
    record: AutonomousTask,
    *,
    credential_handle: CredentialHandle,
  ) -> str:
    if record.event_channel is None:
      raise RuntimeError(
        "autonomous launch envelope requires an owned event channel"
      )
    authority = getattr(
      self,
      "_claim_signing_authority",
      None,
    )
    if type(authority) is not GatewayClaimSigningAuthority:
      raise RuntimeError(
        "autonomous dispatch requires installed claim-signing authority"
      )
    tenant_id = str(getattr(self, "_tenant_id", "") or "").strip()
    if not tenant_id:
      raise RuntimeError(
        "tenant_id is required for autonomous session authority"
      )
    capability_bind = record.capability_bind
    if capability_bind is None:
      raise RuntimeError(
        "autonomous launch envelope requires a capability binding"
      )
    if (
      type(credential_handle) is not CredentialHandle
      or credential_handle.handle_id
      != capability_bind.credential_ref
      or credential_handle.principal
      != capability_bind.credential_principal
      or credential_handle.tenant_id != tenant_id
      or credential_handle.provider != capability_bind.provider
      or (
        credential_handle.principal == "user"
        and credential_handle.actor_id
        != (record.owner_user_id or record.user_id)
      )
      or (
        credential_handle.principal == "service"
        and credential_handle.actor_id is not None
      )
    ):
      raise RuntimeError(
        "autonomous session authority requires the exact selected "
        "credential handle"
      )
    created_at = int(record.started_at)
    session_authority = AutonomousSessionAuthority.ordinary(
      OrdinaryAutonomousSessionAuthority(
        session_id=record.task_id,
        tenant_id=tenant_id,
        user_id=record.owner_user_id or record.user_id,
        owner_user_id=record.owner_user_id or record.user_id,
        created_at=created_at,
        expires_at=(
          created_at + _AUTONOMOUS_SESSION_AUTHORITY_TTL_SECONDS
        ),
        user_email=record.user_email,
        risk_user_id=record.risk_user_id,
        role=record.role,
        kind="chat",
        channel=record.channel or "cli",
        purpose=AUTONOMOUS_RUNTIME_SESSION_PURPOSE,
        raw_user_id=record.raw_user_id,
        user_slug=record.user_slug,
        user_aliases=tuple(record.user_aliases),
        identity_status=record.identity_status,
        schema_version=DEFAULT_SCHEMA_VERSION,
        is_public=False,
        allow_service_for_interactive=False,
        auth_provider=capability_bind.provider,
        credential_handle=credential_handle,
      ),
      dispatch_scope=(
        AutonomousDispatchScope.from_mapping(record.dispatch_scope)
        if record.dispatch_scope is not None
        else None
      ),
    )
    workload = AutonomousLaunchWorkload(
      profile=record.profile,
      mode="run_once" if record.mode == "once" else record.mode,
      task=record.task,
      skill=record.skill,
      pack=record.pack,
      context=record.context,
      ticker=record.ticker,
      dev_mode=False,
      max_budget_usd=record.max_budget_usd,
      deliver=record.deliver,
    )
    control_authority = record.control_authority
    if type(control_authority) is not AutonomousControlAuthority:
      raise RuntimeError(
        "autonomous launch envelope requires exact control authority"
      )
    return authority.sign_autonomous_launch_envelope(
      task_id=record.task_id,
      control_run_id=record.control_run_id,
      owner_user_id=record.owner_user_id or record.user_id,
      channel_id=record.event_channel.channel_id,
      bind=capability_bind,
      workload=workload,
      control_authority=control_authority,
      session_authority=session_authority,
    )

  @_with_run_mutation_lock
  async def start(
    self,
    *,
    profile: str,
    mode: str,
    user_id: str,
    user_email: str | None,
    role: str,
    owner_user_id: str | None = None,
    user_slug: str | None = None,
    risk_user_id: int | None = None,
    user_aliases: list[str] | tuple[str, ...] | None = None,
    identity_status: str | None = None,
    control_run_id: str | None = None,
    task: str | None = None,
    skill: str | None = None,
    pack: str | None = None,
    deliver: bool = True,
    context: str | None = None,
    ticker: str | None = None,
    max_budget_usd: float | None = None,
    channel: str | None = None,
    dispatch_scope: dict[str, Any] | None = None,
    resumed_from: str | None = None,
    schedule_id: str | None = None,
    schedule_name: str | None = None,
  ) -> dict[str, Any]:
    normalized_role = require_exact_role(role)
    await self._reserve_slot()
    task_id = self._next_task_id()
    log_handle = None
    record: AutonomousTask | None = None
    event_channel_parent = None
    event_channel_child: AutonomousEventChannelChild | None = None
    approval_channel_child: AutonomousApprovalChannelChild | None = None
    claim_broker_child_fd = -1
    approval_channel_child_fd = -1
    owner_lifeline_read_fd = -1
    owner_lifeline_write_fd = -1
    owner_lease_fd = -1
    created_run_files: list[tuple[Path, int, int]] = []
    ownership_transferred = False
    try:
      authority = getattr(
        self,
        "_claim_signing_authority",
        None,
      )
      if type(authority) is not GatewayClaimSigningAuthority:
        raise RuntimeError(
          "autonomous dispatch requires installed claim-signing authority"
        )
      if control_run_id is None:
        control_run_id = task_id
      elif (
        type(control_run_id) is not str
        or not control_run_id.strip()
        or len(control_run_id.strip()) > 512
        or any(
          ord(character) < 0x20
          for character in control_run_id.strip()
        )
      ):
        raise ValueError(
          "control_run_id must be a canonical non-empty string"
        )
      else:
        control_run_id = control_run_id.strip()
      raw_user_id = str(user_id or "").strip()
      if not raw_user_id:
        raise ValueError("user_id is required")
      identity = _start_identity_payload(
        raw_user_id=raw_user_id,
        user_email=user_email,
        owner_user_id=owner_user_id,
        user_slug=user_slug,
        risk_user_id=risk_user_id,
        user_aliases=user_aliases,
        identity_status=identity_status,
      )
      normalized_owner_user_id = str(identity["owner_user_id"])
      normalized_user_slug = identity["user_slug"]
      normalized_risk_user_id = int(identity["risk_user_id"])
      normalized_identity_status = str(identity["identity_status"])
      aliases = list(identity["user_aliases"])
      normalized_dispatch_scope = _normalize_dispatch_scope(dispatch_scope)
      if dispatch_scope is not None and normalized_dispatch_scope is None:
        raise ValueError("dispatch_scope must be a redacted portfolio dispatch scope")
      normalized_max_budget_usd = normalize_max_budget_usd(max_budget_usd)
      normalize_profile = _runtime_attr(
        "normalize_autonomous_profile",
        normalize_autonomous_profile,
      )
      normalized_profile = normalize_profile(profile)
      normalized_mode = mode.strip().lower()
      if normalized_mode not in {"once", "task", "skill", "pack"}:
        raise ValueError("mode must be once, task, skill, or pack")
      normalized_skill = (
        skill.strip() if isinstance(skill, str) and skill.strip() else None
      )
      if pack is not None and not isinstance(pack, str):
        raise ValueError("pack must be a string")
      normalized_pack = (
        pack.strip() if isinstance(pack, str) and pack.strip() else None
      )
      if not isinstance(deliver, bool):
        raise ValueError("deliver must be a bool")
      normalized_deliver = deliver
      normalized_channel = (
        channel.strip().lower()
        if isinstance(channel, str) and channel.strip()
        else None
      )
      normalized_resumed_from = (
        resumed_from.strip()
        if isinstance(resumed_from, str) and resumed_from.strip()
        else None
      )
      normalized_schedule_id = (
        schedule_id.strip()
        if isinstance(schedule_id, str) and schedule_id.strip()
        else None
      )
      if normalized_resumed_from is not None and normalized_schedule_id is not None:
        raise ValueError("autonomous launch cannot be both a resume and a schedule fire")
      resumed_record = None
      if normalized_resumed_from is not None:
        resumed_record = self._find_by_control_run_id(normalized_resumed_from)
        if resumed_record is None:
          raise ValueError(
            f"Unknown resumed_from control_run_id: {normalized_resumed_from}"
          )
        if (
          resumed_record.owner_user_id or resumed_record.user_id
        ) != normalized_owner_user_id:
          raise PermissionError("resumed autonomous run owner does not match launch owner")
        self._require_released_owner_lease(resumed_record)
        normalized_pack = resumed_record.pack
        normalized_deliver = resumed_record.deliver

      cmd = self._build_cmd(
        profile=normalized_profile,
        mode=normalized_mode,
        task=task,
        skill=normalized_skill,
        pack=normalized_pack,
        deliver=normalized_deliver,
        context=context,
        ticker=ticker,
        max_budget_usd=normalized_max_budget_usd,
      )
      os_module = _runtime_attr("os", os)

      binding_source = "start"
      binding_run_mode = "autonomous"
      required_bind = None
      if normalized_resumed_from is not None:
        binding_source = "resume"
        assert resumed_record is not None
        required_bind = resumed_record.capability_bind
        if required_bind is None:
          raise RuntimeError(
            "resumed autonomous run has no persisted capability binding"
          )
        binding_run_mode = required_bind.run_mode
      elif normalized_schedule_id is not None:
        binding_source = "schedule"
        binding_run_mode = "cron"

      capability_binding = await resolve_autonomous_capability_binding(
        self._autonomous_capability_binding_resolver,
        AutonomousCapabilityBindingRequest(
          task_id=task_id,
          control_run_id=control_run_id,
          owner_user_id=normalized_owner_user_id,
          raw_user_id=raw_user_id,
          user_email=user_email,
          profile=normalized_profile,
          mode=normalized_mode,
          skill=normalized_skill,
          channel=normalized_channel,
          source=binding_source,
          run_mode=binding_run_mode,
          required_bind=required_bind,
        ),
      )
      credential_handoff_payload = (
        encode_autonomous_credential_handoff(
          capability_binding.materialized_credential
        )
      )

      self._log_dir.mkdir(parents=True, exist_ok=True)
      canonical_log_dir = self._log_dir.resolve()
      log_path = canonical_log_dir / f"{task_id}.log"
      operator_inbox_path = (
        canonical_log_dir / f"{task_id}.operator-messages.jsonl"
      )
      events_path = canonical_log_dir / f"{task_id}.events.jsonl"
      approval_decisions_path = None
      owner_lease_path = (
        canonical_log_dir / f"{task_id}.owner-lease"
      )
      admission_identity = prepare_autonomous_admission_ledger(
        canonical_log_dir / ".autonomous-admission-ledger.sqlite3"
      )

      operator_fd, operator_stat = secure_create_owned_file(
        operator_inbox_path
      )
      created_run_files.append((
        operator_inbox_path,
        operator_stat.st_dev,
        operator_stat.st_ino,
      ))
      os_module.close(operator_fd)

      events_fd, events_stat = secure_create_owned_file(events_path)
      created_run_files.append((
        events_path,
        events_stat.st_dev,
        events_stat.st_ino,
      ))
      os_module.close(events_fd)
      fsync_owned_file_directory(events_path)

      owner_lease_fd, owner_lease_stat = secure_create_owned_file(
        owner_lease_path
      )
      created_run_files.append((
        owner_lease_path,
        owner_lease_stat.st_dev,
        owner_lease_stat.st_ino,
      ))
      fcntl.flock(
        owner_lease_fd,
        fcntl.LOCK_EX | fcntl.LOCK_NB,
      )

      control_authority = AutonomousControlAuthority(
        control_mode="file",
        admission_ledger_path=admission_identity.path,
        admission_ledger_device=admission_identity.device,
        admission_ledger_inode=admission_identity.inode,
        operator_inbox_path=str(operator_inbox_path),
        operator_inbox_device=operator_stat.st_dev,
        operator_inbox_inode=operator_stat.st_ino,
        approval_decisions_path=None,
        approval_decisions_device=None,
        approval_decisions_inode=None,
        approval_store_path=None,
        approval_store_device=None,
        approval_store_inode=None,
      )

      log_fd, log_stat = secure_create_owned_file(log_path)
      created_run_files.append((
        log_path,
        log_stat.st_dev,
        log_stat.st_ino,
      ))
      log_handle = os_module.fdopen(log_fd, "wb")
      event_channel_pair = create_autonomous_event_channel(
        io_timeout_seconds=AUTONOMOUS_EVENT_CHANNEL_MAX_IO_TIMEOUT_SECONDS,
      )
      event_channel_parent = event_channel_pair.parent
      event_channel_child = event_channel_pair.child
      (
        owner_lifeline_read_fd,
        owner_lifeline_write_fd,
      ) = _create_owner_lifeline()
      time_module = _runtime_attr("time", time)
      record = AutonomousTask(
        task_id=task_id,
        control_run_id=control_run_id,
        session_id=task_id,
        channel_id=event_channel_parent.channel_id,
        user_id=normalized_owner_user_id,
        user_email=user_email,
        role=normalized_role,
        profile=normalized_profile,
        mode=normalized_mode,
        task=task.strip() if isinstance(task, str) and task.strip() else None,
        skill=normalized_skill,
        pack=normalized_pack,
        deliver=normalized_deliver,
        context=context.strip() if isinstance(context, str) and context.strip() else None,
        ticker=canonicalize_ticker(ticker) if isinstance(ticker, str) and ticker.strip() else None,
        channel=normalized_channel,
        dev_mode=False,
        dispatch_scope=normalized_dispatch_scope,
        cmd=cmd,
        log_path=log_path,
        events_path=events_path,
        events_device=events_stat.st_dev,
        events_inode=events_stat.st_ino,
        events_evidence_status="complete",
        operator_inbox_path=operator_inbox_path,
        approval_decisions_path=approval_decisions_path,
        control_authority=control_authority,
        owner_lease_path=owner_lease_path,
        owner_lease_device=owner_lease_stat.st_dev,
        owner_lease_inode=owner_lease_stat.st_ino,
        started_at=time_module.time(),
        max_budget_usd=normalized_max_budget_usd,
        state="starting",
        log_handle=log_handle,
        slot_reserved=True,
        event_lines=[],
        event_channel=event_channel_parent,
        owner_lifeline_fd=owner_lifeline_write_fd,
        owner_user_id=normalized_owner_user_id,
        raw_user_id=raw_user_id,
        user_slug=normalized_user_slug,
        risk_user_id=normalized_risk_user_id,
        user_aliases=aliases,
        identity_status=normalized_identity_status,
        resumed_from=normalized_resumed_from,
        schedule_id=normalized_schedule_id,
        schedule_name=schedule_name.strip() if isinstance(schedule_name, str) and schedule_name.strip() else None,
        tool_result_spill_dir=self._expected_tool_result_spill_dir(task_id),
        capability_bind=capability_binding.bind,
      )
      owner_lifeline_write_fd = -1
      self._attach_manifest_tracking(record)
      self._tasks[task_id] = record
      asyncio_module = _asyncio_module()

      if not self._write_task_manifest(record, checked=True):
        record.tool_result_spill_dir = None
        raise RuntimeError("failed to persist starting autonomous task manifest")
      try:
        record.tool_result_spill_dir.mkdir(mode=0o700)
      except Exception as exc:
        raise RuntimeError(f"autonomous spill directory setup failed: {exc}") from exc

      env = _positive_autonomous_child_env(
        os_module.environ,
        provider=capability_binding.bind.provider,
        profile=record.profile,
        deliver=record.deliver,
      )
      for retired_name in _RETIRED_AUTONOMOUS_CHILD_ENV_NAMES:
        env.pop(retired_name, None)
      env["PYTHONUNBUFFERED"] = "1"
      envelope_json = self._signed_launch_envelope(
        record,
        credential_handle=(
          capability_binding.materialized_credential.handle
        ),
      )
      verified_envelope = authority.verify_autonomous_launch_envelope(
        envelope_json
      )
      child_session = verified_envelope.session_authority.to_gateway_session()
      if (
        verified_envelope.task_id != record.task_id
        or verified_envelope.control_run_id
        != record.control_run_id
        or verified_envelope.channel_id != record.channel_id
        or child_session.session_id != record.session_id
      ):
        raise RuntimeError(
          "signed autonomous launch envelope changed task authority"
        )
      env["GATEWAY_USER_KEYS"] = _narrowed_mcp_gateway_user_keys(
        api_dir=self._api_dir,
        user_id=child_session.user_id,
        user_email=child_session.user_email,
      )
      record.launch_nonce = verified_envelope.nonce
      if self._approval_store is not None:
        approval_channel_pair = create_autonomous_approval_channel(
          authority=AutonomousApprovalChannelAuthority(
            launch_nonce=verified_envelope.nonce,
            task_id=record.task_id,
            control_run_id=record.control_run_id,
            session_id=record.session_id,
            channel_id=record.channel_id,
          ),
        )
        record.approval_channel = approval_channel_pair.parent
        approval_channel_child = approval_channel_pair.child
        approval_channel_child_fd = (
          approval_channel_child.take_inherited_fd()
        )
        approval_channel_child = None
        if approval_channel_child_fd < 0:
          raise RuntimeError(
            "autonomous approval channel child endpoint "
            "closed before spawn"
          )
        env[AUTONOMOUS_APPROVAL_CHANNEL_FD_ENV] = str(
          approval_channel_child_fd
        )
      env[AUTONOMOUS_CAPABILITY_ENVELOPE_ENV] = envelope_json
      record.claim_broker = AutonomousClaimBroker(
        authority,
        envelope_json,
      )
      claim_broker_child_fd = (
        record.claim_broker.take_child_fd()
      )
      env[AUTONOMOUS_CLAIM_BROKER_FD_ENV] = str(
        claim_broker_child_fd
      )
      child_event_channel_fd = event_channel_child.fileno()
      if child_event_channel_fd < 0:
        raise RuntimeError(
          "autonomous event channel child endpoint closed before spawn"
        )
      env[AUTONOMOUS_EVENT_CHANNEL_FD_ENV] = str(child_event_channel_fd)
      if record.tool_result_spill_dir is not None:
        env["AGENT_AUTONOMOUS_TOOL_RESULT_SPILL_DIR"] = str(record.tool_result_spill_dir)
      env[AUTONOMOUS_CREDENTIAL_HANDOFF_ENV] = (
        AUTONOMOUS_CREDENTIAL_HANDOFF_STDIN
      )
      try:
        await self._spawn_owned_process(
          record,
          asyncio_module.create_subprocess_exec(
            *self._owned_process_sentinel_cmd(
              cmd,
              owner_lifeline_fd=owner_lifeline_read_fd,
              owner_lease_fd=owner_lease_fd,
            ),
            cwd=str(self._api_dir),
            stdin=asyncio_module.subprocess.PIPE,
            stdout=log_handle,
            stderr=asyncio_module.subprocess.PIPE,
            env=env,
            pass_fds=(
              child_event_channel_fd,
              claim_broker_child_fd,
              *(
                (approval_channel_child_fd,)
                if approval_channel_child_fd >= 0
                else ()
              ),
              owner_lifeline_read_fd,
              owner_lease_fd,
            ),
            start_new_session=True,
          ),
        )
      finally:
        event_channel_child.close()
        event_channel_child = None
        if claim_broker_child_fd >= 0:
          os_module.close(claim_broker_child_fd)
          claim_broker_child_fd = -1
      record.event_channel_task = asyncio_module.create_task(
        self._drain_event_channel(task_id)
      )
      credential_stdin = getattr(record.proc, "stdin", None)
      if credential_stdin is None:
        raise RuntimeError(
          "autonomous credential handoff pipe is unavailable"
        )
      try:
        credential_stdin.write(credential_handoff_payload)
        await credential_stdin.drain()
      finally:
        credential_stdin.close()
        wait_closed = getattr(credential_stdin, "wait_closed", None)
        if callable(wait_closed):
          await wait_closed()

      assert record is not None
      record.state = "running"
      if not self._write_task_manifest(record, checked=True):
        raise RuntimeError("failed to commit running autonomous task manifest")
      record.reaper_task = asyncio_module.create_task(self._reap(task_id))
      await self._publish_run_state(record, "running")
      ownership_transferred = True
      return self._start_payload(record)
    except OSError as exc:
      raise RuntimeError(f"spawn failed: {exc}") from exc
    finally:
      if not ownership_transferred:
        await self._await_cleanup(
          self._cleanup_uncommitted_start(
            task_id=task_id,
            record=record,
            log_handle=log_handle,
            event_channel_child=event_channel_child,
            approval_channel_child=approval_channel_child,
          )
        )
        if record is None and event_channel_parent is not None:
          event_channel_parent.close()
      for owned_fd in (
        owner_lifeline_read_fd,
        owner_lifeline_write_fd,
        owner_lease_fd,
        claim_broker_child_fd,
        approval_channel_child_fd,
      ):
        if owned_fd >= 0:
          os.close(owned_fd)
      if not ownership_transferred:
        for created_path, created_device, created_inode in reversed(
          created_run_files
        ):
          unlink_created_owned_file(
            created_path,
            device=created_device,
            inode=created_inode,
          )


__all__ = [
  "AutonomousRegistryStartMixin",
  "_SPAWN_CLEANUP_GRACE_SEC",
  "_get_process_group_id",
  "_signal_process_group",
]
