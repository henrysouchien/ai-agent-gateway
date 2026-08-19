from __future__ import annotations

import asyncio
import io
import json
import os
import signal
from itertools import count
from pathlib import Path

import pytest

from agent_gateway import AnthropicProvider
import agent_gateway.autonomous_credential_handoff as credential_handoff
from agent_gateway.autonomous_capability_handoff import (
  AutonomousCapabilityBinding,
  AutonomousCapabilityBindingRequest,
  resolve_autonomous_capability_binding,
)
from agent_gateway.autonomous_credential_handoff import (
  AUTONOMOUS_CREDENTIAL_HANDOFF_ENV,
  AUTONOMOUS_CREDENTIAL_HANDOFF_STDIN,
  read_autonomous_credential_handoff,
)
from agent_gateway.autonomous_launch_envelope import (
  AUTONOMOUS_CAPABILITY_ENVELOPE_ENV,
  AUTONOMOUS_TASK_ID_ENV,
  verify_autonomous_launch_envelope,
)
from agent_gateway.autonomous_runner import AutonomousRegistry
from agent_gateway.autonomous_runner_start import (
  _positive_autonomous_child_env,
)
from agent_gateway.capability_binding import CapabilityBind, CredentialHandle
from agent_gateway.capability_execution import MaterializedCredential
from agent_gateway.claim_signing_authority import (
  GatewayClaimSigningAuthority,
)


_SECRET = "autonomous-handoff-test-secret-at-least-32-bytes"
_API_DIR = Path(__file__).resolve().parents[3] / "api"
_FAKE_PROCESS_PIDS = count(110_000)
_FAKE_PROCESSES: dict[int, "_FakeProcess"] = {}


class _FakeProcess:
  def __init__(self, *, stdin=None) -> None:
    self.pid = next(_FAKE_PROCESS_PIDS)
    self.stdin = stdin if stdin is not None else _FakeStdin()
    self.returncode: int | None = None
    _FAKE_PROCESSES[self.pid] = self

  async def wait(self) -> int:
    self.returncode = 0
    return 0

  def terminate(self) -> None:
    self.returncode = -15

  def kill(self) -> None:
    self.returncode = -9


@pytest.fixture(autouse=True)
def _fake_process_groups(monkeypatch):
  from agent_gateway import autonomous_runner

  monkeypatch.setenv(
    "GATEWAY_USER_KEYS",
    json.dumps([{
      "key": "test-mcp-key",
      "slug": "",
      "email": "owner@example.com",
      "risk_user_id": 42,
      "channel": "mcp",
      "role": "owner",
    }]),
  )

  def fake_getpgid(pid: int) -> int:
    if pid not in _FAKE_PROCESSES:
      raise ProcessLookupError(pid)
    return pid

  def fake_killpg(pid: int, signal_number: int) -> None:
    process = _FAKE_PROCESSES.get(pid)
    if process is None:
      raise ProcessLookupError(pid)
    if process.returncode is not None:
      return
    if signal_number == signal.SIGKILL:
      process.kill()
    else:
      process.terminate()

  monkeypatch.setattr(
    autonomous_runner,
    "_get_process_group_id",
    fake_getpgid,
  )
  monkeypatch.setattr(
    autonomous_runner,
    "_signal_process_group",
    fake_killpg,
  )


class _FakeStdin:
  def __init__(self) -> None:
    self.buffer = bytearray()
    self.drain_calls = 0
    self.closed = False
    self.wait_closed_calls = 0

  def write(self, payload: bytes) -> None:
    self.buffer.extend(payload)

  async def drain(self) -> None:
    self.drain_calls += 1

  def close(self) -> None:
    self.closed = True

  async def wait_closed(self) -> None:
    self.wait_closed_calls += 1


def _bind(
  *,
  run_mode: str,
  model: str = "claude-test",
  principal: str = "service",
  credential_ref: str | None = None,
) -> CapabilityBind:
  return CapabilityBind(
    schema_version="1.0",
    capability_id="session.driver",
    model_key=f"anthropic.test-{model}",
    provider="anthropic",
    upstream_model=model,
    adapter="anthropic.messages",
    protocol_profile="messages.standard",
    route="anthropic.public",
    effort="medium",
    credential_principal=principal,  # type: ignore[arg-type]
    credential_ref=(
      credential_ref
      or (
        "autonomous-user:exact-handle"
        if principal == "user"
        else _service_handle().handle_id
      )
    ),
    run_mode=run_mode,  # type: ignore[arg-type]
    registry_revision="test-registry-1",
    policy_revision="test-policy-1",
    selection_source="capability_default",
  )


def _service_handle() -> CredentialHandle:
  return CredentialHandle(
    handle_id="service:test-product:anthropic",
    provider="anthropic",
    principal="service",
    tenant_id="test-product",
    actor_id=None,
  )


def _service_binding(
  bind: CapabilityBind,
) -> AutonomousCapabilityBinding:
  handle = _service_handle()
  return AutonomousCapabilityBinding(
    bind=bind,
    materialized_credential=MaterializedCredential(
      handle=handle,
      auth_config={
        "provider": handle.provider,
        "auth_mode": "api",
        "api_key": "service-secret",
      },
    ),
  )


def _binding_for_request(request: AutonomousCapabilityBindingRequest):
  return _service_binding(
    request.required_bind or _bind(run_mode=request.run_mode),
  )


def _registry(tmp_path: Path, resolver=_binding_for_request) -> AutonomousRegistry:
  return AutonomousRegistry(
    api_dir=_API_DIR,
    tenant_id="test-product",
    python_executable="python3",
    log_dir=tmp_path,
    max_running=4,
    service_provider_handles={
      "anthropic": _service_handle(),
    },
    autonomous_capability_binding_resolver=resolver,
    claim_signing_authority=GatewayClaimSigningAuthority(_SECRET),
  )


async def _start(
  registry: AutonomousRegistry,
  *,
  resumed_from: str | None = None,
  schedule_id: str | None = None,
  mode: str = "skill",
  skill: str | None = "risk.scan",
  pack: str | None = None,
  deliver: bool = True,
  role: str = "owner",
) -> dict:
  payload = await registry.start(
    role=role,
    profile="analyst",
    mode=mode,
    skill=skill,
    pack=pack,
    deliver=deliver,
    user_id="42",
    user_email="owner@example.com",
    owner_user_id="42",
    resumed_from=resumed_from,
    schedule_id=schedule_id,
    schedule_name="Daily risk" if schedule_id else None,
  )
  await registry.wait(payload["task_id"], timeout_sec=1)
  return payload


def test_resolver_is_invoked_once_and_signed_exact_envelope_reaches_spawn(
  monkeypatch,
  tmp_path,
) -> None:
  requests: list[AutonomousCapabilityBindingRequest] = []
  captured_env: dict[str, str] = {}
  service_stdin = _FakeStdin()

  async def resolver(request: AutonomousCapabilityBindingRequest):
    requests.append(request)
    return _binding_for_request(request)

  async def fake_exec(*args, **kwargs):
    _ = args
    captured_env.update(kwargs["env"])
    return _FakeProcess(stdin=service_stdin)

  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", _SECRET)
  monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
  registry = _registry(tmp_path, resolver)

  payload = asyncio.run(_start(registry))

  assert len(requests) == 1
  request = requests[0]
  assert request.source == "start"
  assert request.run_mode == "autonomous"
  assert request.profile == "analyst"
  assert request.skill == "risk.scan"
  assert AUTONOMOUS_TASK_ID_ENV not in captured_env
  envelope = verify_autonomous_launch_envelope(
    _SECRET,
    captured_env[AUTONOMOUS_CAPABILITY_ENVELOPE_ENV],
  )
  assert envelope.task_id == payload["task_id"]
  assert envelope.control_run_id == payload["run_id"]
  assert envelope.owner_user_id == "42"
  assert envelope.bind == _bind(run_mode="autonomous")
  assert (
    captured_env[AUTONOMOUS_CREDENTIAL_HANDOFF_ENV]
    == AUTONOMOUS_CREDENTIAL_HANDOFF_STDIN
  )
  materialized = read_autonomous_credential_handoff(
    expected_handle_id=_service_handle().handle_id,
    expected_provider=envelope.bind.provider,
    expected_principal=envelope.bind.credential_principal,
    expected_tenant_id="test-product",
    expected_actor_id=None,
    stream=io.BytesIO(bytes(service_stdin.buffer)),
  )
  assert materialized.handle == _service_handle()
  assert materialized.auth_config["api_key"] == "service-secret"
  record = registry._tasks[payload["task_id"]]
  assert record.capability_bind.credential_ref == materialized.handle.handle_id


def test_invite_role_reaches_record_and_signed_child_authority(
  monkeypatch,
  tmp_path,
) -> None:
  captured_env: dict[str, str] = {}

  async def fake_exec(*args, **kwargs):
    _ = args
    captured_env.update(kwargs["env"])
    return _FakeProcess(stdin=_FakeStdin())

  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", _SECRET)
  monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
  registry = _registry(tmp_path)

  payload = asyncio.run(_start(registry, role="invite"))
  envelope = verify_autonomous_launch_envelope(
    _SECRET,
    captured_env[AUTONOMOUS_CAPABILITY_ENVELOPE_ENV],
  )

  assert registry._tasks[payload["task_id"]].role == "invite"
  assert envelope.session_authority.to_gateway_session().role == "invite"


@pytest.mark.parametrize("role", ["Owner", " owner ", "OWNER", "", None, True])
def test_registry_start_rejects_malformed_role(tmp_path, role: object) -> None:
  registry = _registry(tmp_path)

  async def start() -> None:
    await registry.start(
      role=role,  # type: ignore[arg-type]
      profile="analyst",
      mode="skill",
      skill="risk.scan",
      user_id="42",
      user_email="owner@example.com",
    )

  with pytest.raises(ValueError, match="role must be exactly"):
    asyncio.run(start())
  assert registry._tasks == {}
  assert registry._reserved_slots == 0


def test_user_credential_exact_signed_handle_reaches_child_over_stdin(
  monkeypatch,
  tmp_path,
) -> None:
  handle = CredentialHandle(
    handle_id="autonomous-user:exact-handle",
    provider="anthropic",
    principal="user",
    tenant_id="test-product",
    actor_id="42",
  )
  materialized = MaterializedCredential(
    handle=handle,
    auth_config={
      "provider": "anthropic",
      "auth_mode": "api",
      "api_key": "parent-user-secret",
    },
  )
  user_bind = _bind(
    run_mode="autonomous",
    principal="user",
    credential_ref=handle.handle_id,
  )
  parent_stdin = _FakeStdin()
  captured_env: dict[str, str] = {}
  captured_spawn: dict[str, object] = {}

  async def resolver(
    request: AutonomousCapabilityBindingRequest,
  ) -> AutonomousCapabilityBinding:
    assert request.owner_user_id == "42"
    return AutonomousCapabilityBinding(
      bind=user_bind,
      materialized_credential=materialized,
    )

  async def fake_exec(*args, **kwargs):
    captured_spawn["args"] = args
    captured_spawn["stdin"] = kwargs["stdin"]
    captured_env.update(kwargs["env"])
    return _FakeProcess(stdin=parent_stdin)

  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", _SECRET)
  monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
  registry = _registry(tmp_path, resolver)

  payload = asyncio.run(_start(registry))

  assert captured_spawn["stdin"] == asyncio.subprocess.PIPE
  assert (
    captured_env[AUTONOMOUS_CREDENTIAL_HANDOFF_ENV]
    == AUTONOMOUS_CREDENTIAL_HANDOFF_STDIN
  )
  assert "parent-user-secret" not in json.dumps(captured_env)
  envelope = verify_autonomous_launch_envelope(
    _SECRET,
    captured_env[AUTONOMOUS_CAPABILITY_ENVELOPE_ENV],
  )
  assert envelope.task_id == payload["task_id"]
  assert envelope.control_run_id == payload["run_id"]
  assert envelope.owner_user_id == "42"
  assert envelope.bind == user_bind
  assert envelope.bind.credential_ref == handle.handle_id
  assert parent_stdin.drain_calls == 1
  assert parent_stdin.closed is True
  assert parent_stdin.wait_closed_calls == 1

  child_materialized = read_autonomous_credential_handoff(
    expected_handle_id=envelope.bind.credential_ref,
    expected_provider=envelope.bind.provider,
    expected_principal=envelope.bind.credential_principal,
    expected_tenant_id=handle.tenant_id,
    expected_actor_id=envelope.owner_user_id,
    stream=io.BytesIO(bytes(parent_stdin.buffer)),
  )
  assert child_materialized.handle == handle
  assert child_materialized.auth_config == materialized.auth_config
  record = registry._tasks[payload["task_id"]]
  assert record.capability_bind == envelope.bind
  manifest_text = (
    tmp_path / f"{payload['task_id']}.task.json"
  ).read_text(encoding="utf-8")
  assert "parent-user-secret" not in manifest_text
  assert json.loads(manifest_text)["capability_bind"] == envelope.bind.receipt()


def test_child_closes_inherited_user_credential_pipe(
  monkeypatch,
) -> None:
  handle = CredentialHandle(
    handle_id="autonomous-user:child-close",
    provider="anthropic",
    principal="user",
    tenant_id="test-product",
    actor_id="42",
  )
  materialized = MaterializedCredential(
    handle=handle,
    auth_config={
      "provider": "anthropic",
      "auth_mode": "api",
      "api_key": "child-close-secret",
    },
  )
  inherited_stdin = io.BytesIO(
    credential_handoff.encode_autonomous_credential_handoff(
      materialized
    )
  )
  monkeypatch.setattr(
    credential_handoff.sys,
    "stdin",
    type("_TestStdin", (), {"buffer": inherited_stdin})(),
  )

  decoded = read_autonomous_credential_handoff(
    expected_handle_id=handle.handle_id,
    expected_provider=handle.provider,
    expected_principal=handle.principal,
    expected_tenant_id=handle.tenant_id,
    expected_actor_id=handle.actor_id or "",
  )

  assert decoded.handle == handle
  assert decoded.auth_config == materialized.auth_config
  assert inherited_stdin.closed is True


def test_child_environment_is_profile_scoped_and_secret_minimal() -> None:
  source = {
    "PATH": "/usr/bin",
    "OPENAI_MODEL": "gpt-test",
    "ANTHROPIC_MODEL": "claude-test",
    "OPENAI_API_KEY": "openai-secret",
    "ANTHROPIC_API_KEY": "anthropic-secret",
    "CODEX_AUTH_JSON": "codex-secret",
    "XAI_API_KEY": "xai-secret",
    "GATEWAY_USER_KEYS": "gateway-user-secret",
    "AGENT_API_USER_CLAIM_HMAC_KEY": "signing-secret",
    "DATABASE_URL": "postgres://secret",
    "AWS_ACCESS_KEY_ID": "aws-id",
    "AWS_SECRET_ACCESS_KEY": "aws-secret",
    "KMS_KEY_ID": "kms-secret",
    "GMAIL_CREDENTIALS_JSON": "gmail-secret",
    "OFFICE_ADDIN_CLIENT_SECRET": "addin-secret",
    "AGENTS_API_KEY": "agents-secret",
    "FMP_API_KEY": "research-secret",
    "IBKR_FLEX_TOKEN": "brokerage-secret",
    "TELEGRAM_BOT_TOKEN": "telegram-secret",
    "TELEGRAM_CHAT_ID": "telegram-chat",
    "AGENT_GATEWAY_RATES_FILE": "/opt/hank/rates.json",
    "AGENT_SESSION_LOG_BASE_DIR": "/writable/live/api/sessions",
    "CORPUS_LOG_DIR": "/writable/live/corpus-logs",
    "CORPUS_STATE_DIR": "/writable/live/corpus",
    "EDGAR_UPDATER_ROOT": "/opt/hank/edgar-updater",
    "LOCAL_GATEWAY_CONTROL_GENERATION": "/opt/hank/control/generation",
    "LOCAL_GATEWAY_CONTROL_GENERATION_ID": "generation-1",
    "LOG_DIR": "/writable/live/logs",
    "MCP_STARTUP_CONCURRENCY": "1",
    "MCP_STDIO_CONNECT_BACKOFF_S": "3",
    "MCP_STDIO_CONNECT_RETRIES": "6",
    "MCP_STDIO_CONNECT_STABILIZE_S": "0.5",
    "OPENAI_SESSION_EPOCH": "responses-v1",
  }
  forbidden = {
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "CODEX_AUTH_JSON",
    "XAI_API_KEY",
    "GATEWAY_USER_KEYS",
    "AGENT_API_USER_CLAIM_HMAC_KEY",
    "DATABASE_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "KMS_KEY_ID",
    "GMAIL_CREDENTIALS_JSON",
    "OFFICE_ADDIN_CLIENT_SECRET",
    "AGENTS_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
  }

  analyst = _positive_autonomous_child_env(
    source,
    provider="openai",
    profile="analyst",
    deliver=False,
  )
  assert "OPENAI_MODEL" not in analyst
  assert analyst["AGENT_GATEWAY_RATES_FILE"] == "/opt/hank/rates.json"
  assert analyst["FMP_API_KEY"] == "research-secret"
  # Session-log root must reach the child or it falls back to a path inside
  # the immutable promoted snapshot and dies on mkdir (bg_76/bg_77 2026-07-31).
  assert analyst["AGENT_SESSION_LOG_BASE_DIR"] == "/writable/live/api/sessions"
  assert analyst["CORPUS_LOG_DIR"] == "/writable/live/corpus-logs"
  assert analyst["CORPUS_STATE_DIR"] == "/writable/live/corpus"
  assert analyst["EDGAR_UPDATER_ROOT"] == "/opt/hank/edgar-updater"
  assert analyst["LOCAL_GATEWAY_CONTROL_GENERATION"] == "/opt/hank/control/generation"
  assert analyst["LOCAL_GATEWAY_CONTROL_GENERATION_ID"] == "generation-1"
  assert analyst["LOG_DIR"] == "/writable/live/logs"
  assert analyst["MCP_STARTUP_CONCURRENCY"] == "1"
  assert analyst["MCP_STDIO_CONNECT_BACKOFF_S"] == "3"
  assert analyst["MCP_STDIO_CONNECT_RETRIES"] == "6"
  assert analyst["MCP_STDIO_CONNECT_STABILIZE_S"] == "0.5"
  # Epoch must reach durable OpenAI children or scope_provider_session_id
  # fails closed at the history fence (OpenAISessionEpochError).
  assert analyst["OPENAI_SESSION_EPOCH"] == "responses-v1"
  assert "ANTHROPIC_MODEL" not in analyst
  assert "IBKR_FLEX_TOKEN" not in analyst
  assert forbidden.isdisjoint(analyst)

  advisor = _positive_autonomous_child_env(
    source,
    provider="openai",
    profile="advisor",
    deliver=True,
  )
  assert advisor["FMP_API_KEY"] == "research-secret"
  assert advisor["AGENT_GATEWAY_RATES_FILE"] == "/opt/hank/rates.json"
  assert advisor["AGENT_SESSION_LOG_BASE_DIR"] == "/writable/live/api/sessions"
  assert advisor["IBKR_FLEX_TOKEN"] == "brokerage-secret"
  assert advisor["ADVISOR_TELEGRAM_BOT_TOKEN"] == "telegram-secret"
  assert advisor["ADVISOR_TELEGRAM_CHAT_ID"] == "telegram-chat"
  assert advisor["OPENAI_SESSION_EPOCH"] == "responses-v1"
  assert "OPENAI_MODEL" not in advisor
  assert forbidden.isdisjoint(advisor)

  custom_provider = _positive_autonomous_child_env(
    source,
    provider="custom",
    profile="custom",
    deliver=True,
  )
  assert "FMP_API_KEY" not in custom_provider
  assert custom_provider["AGENT_GATEWAY_RATES_FILE"] == "/opt/hank/rates.json"
  assert "IBKR_FLEX_TOKEN" not in custom_provider
  assert "OPENAI_SESSION_EPOCH" not in custom_provider
  assert not any("TELEGRAM" in name for name in custom_provider)
  assert forbidden.isdisjoint(custom_provider)


def test_non_anthropic_child_rates_override_reaches_provider_construction(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  rates_path = tmp_path / "rates.json"
  rates_path.write_text(json.dumps({
    "version": "child-projected",
    "source": "https://example.test/rates",
    "providers": {"anthropic": {"models": {}}},
  }), encoding="utf-8")
  projected = _positive_autonomous_child_env(
    {"AGENT_GATEWAY_RATES_FILE": str(rates_path)},
    provider="openai",
    profile="analyst",
    deliver=False,
  )
  monkeypatch.setenv(
    "AGENT_GATEWAY_RATES_FILE",
    projected["AGENT_GATEWAY_RATES_FILE"],
  )

  provider = AnthropicProvider()

  assert provider._rate_table.version == "child-projected"


def test_pinned_autonomous_child_inherits_exact_runtime_import_roots() -> None:
  version_root = Path("/opt/hank/runtime/versions/ai-a_risk-b")

  projected = _positive_autonomous_child_env(
    {"LOCAL_GATEWAY_RUNTIME_VERSION_ROOT": str(version_root)},
    provider="openai",
    profile="research-producer",
    deliver=False,
  )

  ai_root = version_root / "ai-excel-addin"
  risk_root = version_root / "risk_module"
  assert projected["PYTHONPATH"].split(os.pathsep) == [
    str(ai_root),
    str(ai_root / "api"),
    str(ai_root / "packages" / "agent-gateway"),
    str(ai_root / "packages" / "excel-mcp" / "python"),
    str(ai_root / "packages" / "ibkr-relay-client" / "python"),
    str(ai_root / "packages" / "sheets-finance-mcp"),
    str(ai_root / "packages" / "value-semantics-core"),
    str(risk_root / "brokerage-connect"),
    str(risk_root),
  ]


def test_pinned_autonomous_child_preserves_explicit_pythonpath() -> None:
  projected = _positive_autonomous_child_env(
    {
      "LOCAL_GATEWAY_RUNTIME_VERSION_ROOT": "/opt/hank/runtime/version",
      "PYTHONPATH": "/explicit/runtime/path",
    },
    provider="openai",
    profile="research-producer",
    deliver=False,
  )

  assert projected["PYTHONPATH"] == "/explicit/runtime/path"


def test_parent_closes_user_credential_pipe_and_cleans_launch_on_broken_pipe(
  monkeypatch,
  tmp_path,
) -> None:
  handle = CredentialHandle(
    handle_id="autonomous-user:broken-pipe",
    provider="anthropic",
    principal="user",
    tenant_id="test-product",
    actor_id="42",
  )
  user_secret = "broken-pipe-secret-must-not-persist"
  materialized = MaterializedCredential(
    handle=handle,
    auth_config={
      "provider": "anthropic",
      "auth_mode": "api",
      "api_key": user_secret,
    },
  )

  class BrokenPipeStdin(_FakeStdin):
    async def drain(self) -> None:
      self.drain_calls += 1
      raise BrokenPipeError("child closed credential pipe")

  parent_stdin = BrokenPipeStdin()
  process = _FakeProcess(stdin=parent_stdin)

  async def resolver(
    request: AutonomousCapabilityBindingRequest,
  ) -> AutonomousCapabilityBinding:
    return AutonomousCapabilityBinding(
      bind=_bind(
        run_mode=request.run_mode,
        principal="user",
        credential_ref=handle.handle_id,
      ),
      materialized_credential=materialized,
    )

  async def fake_exec(*args, **kwargs):
    _ = args, kwargs
    return process

  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", _SECRET)
  monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
  registry = _registry(tmp_path, resolver)

  with pytest.raises(RuntimeError, match="spawn failed"):
    asyncio.run(_start(registry))

  assert parent_stdin.drain_calls == 1
  assert parent_stdin.closed is True
  assert parent_stdin.wait_closed_calls == 1
  assert registry._tasks == {}
  assert registry._reserved_slots == 0
  assert not list(tmp_path.glob("*.task.json"))
  for path in tmp_path.iterdir():
    if path.is_file():
      assert user_secret not in path.read_text(
        encoding="utf-8",
        errors="replace",
      )


def test_missing_binding_resolver_refuses_before_spawn(monkeypatch, tmp_path) -> None:
  spawn_calls = 0

  async def fake_exec(*args, **kwargs):
    nonlocal spawn_calls
    _ = args, kwargs
    spawn_calls += 1
    return _FakeProcess()

  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", _SECRET)
  monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
  registry = _registry(tmp_path, None)

  with pytest.raises(RuntimeError, match="binding resolver is required"):
    asyncio.run(_start(registry))

  assert spawn_calls == 0
  assert registry._tasks == {}
  assert not list(tmp_path.glob("*.task.json"))


@pytest.mark.parametrize("schedule_id", [None, "schedule-1"])
def test_autonomous_and_cron_refuse_mismatched_credential_binding_before_state_or_spawn(
  monkeypatch,
  tmp_path,
  schedule_id: str | None,
) -> None:
  spawn_calls = 0

  async def resolver(request: AutonomousCapabilityBindingRequest):
    handle = _service_handle()
    return AutonomousCapabilityBinding(
      bind=_bind(
        run_mode=request.run_mode,
        credential_ref="service:different-handle",
      ),
      materialized_credential=MaterializedCredential(
        handle=handle,
        auth_config={"provider": "anthropic", "api_key": "secret"},
      ),
    )

  async def fake_exec(*args, **kwargs):
    nonlocal spawn_calls
    _ = args, kwargs
    spawn_calls += 1
    return _FakeProcess()

  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", _SECRET)
  monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
  registry = _registry(tmp_path, resolver)

  with pytest.raises(
    ValueError,
    match="credential handle does not match",
  ):
    asyncio.run(_start(registry, schedule_id=schedule_id))

  assert spawn_calls == 0
  assert registry._tasks == {}
  assert registry._reserved_slots == 0
  assert not list(tmp_path.glob("*.task.json"))
  assert not list(tmp_path.glob("*.events.jsonl"))
  assert not list(tmp_path.glob("*.log"))


def test_manifest_persists_and_rehydrates_secret_free_exact_handoff(
  monkeypatch,
  tmp_path,
) -> None:
  captured_env: dict[str, str] = {}

  async def fake_exec(*args, **kwargs):
    _ = args
    captured_env.update(kwargs["env"])
    return _FakeProcess()

  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", _SECRET)
  monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
  registry = _registry(tmp_path)
  payload = asyncio.run(_start(registry))
  asyncio.run(registry.wait(payload["task_id"], timeout_sec=1))

  manifest = json.loads(
    (tmp_path / f"{payload['task_id']}.task.json").read_text(encoding="utf-8")
  )
  assert manifest["capability_bind"] == _bind(run_mode="autonomous").receipt()
  assert _SECRET not in json.dumps(manifest)
  assert AUTONOMOUS_CAPABILITY_ENVELOPE_ENV not in manifest

  rehydrated = _registry(tmp_path)
  record = rehydrated._tasks[payload["task_id"]]
  assert record.capability_bind == _bind(run_mode="autonomous")


def test_historical_and_tampered_manifests_are_rejected(
  monkeypatch,
  tmp_path,
) -> None:
  async def fake_exec(*args, **kwargs):
    _ = args, kwargs
    return _FakeProcess()

  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", _SECRET)
  monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
  registry = _registry(tmp_path)
  payload = asyncio.run(_start(registry))
  asyncio.run(registry.wait(payload["task_id"], timeout_sec=1))
  manifest = json.loads(
    (tmp_path / f"{payload['task_id']}.task.json").read_text(encoding="utf-8")
  )

  for version, task_id in ((1, "bg_10"), (2, "bg_11"), (4, "bg_14")):
    historical = {
      **manifest,
      "manifest_version": version,
      "task_id": task_id,
      "control_run_id": task_id,
    }
    (tmp_path / f"{task_id}.task.json").write_text(
      json.dumps(historical) + "\n",
      encoding="utf-8",
    )

  tampered = {
    **manifest,
    "task_id": "bg_12",
    "control_run_id": "bg_12",
    "capability_bind": {
      **manifest["capability_bind"],
      "capability_id": "node.explore",
    },
  }
  (tmp_path / "bg_12.task.json").write_text(
    json.dumps(tampered) + "\n",
    encoding="utf-8",
  )
  invalid_adapter = {
    **manifest,
    "task_id": "bg_13",
    "control_run_id": "bg_13",
    "capability_bind": {
      **manifest["capability_bind"],
      "adapter": "",
    },
  }
  (tmp_path / "bg_13.task.json").write_text(
    json.dumps(invalid_adapter) + "\n",
    encoding="utf-8",
  )

  rehydrated = _registry(tmp_path)
  assert "bg_10" not in rehydrated._tasks
  assert "bg_11" not in rehydrated._tasks
  assert "bg_14" not in rehydrated._tasks
  assert "bg_12" not in rehydrated._tasks
  assert "bg_13" not in rehydrated._tasks


def test_resume_reuses_persisted_exact_bind(
  monkeypatch,
  tmp_path,
) -> None:
  requests: list[AutonomousCapabilityBindingRequest] = []
  envelopes: list[str] = []

  async def resolver(request: AutonomousCapabilityBindingRequest):
    requests.append(request)
    return _binding_for_request(request)

  async def fake_exec(*args, **kwargs):
    _ = args
    envelopes.append(kwargs["env"][AUTONOMOUS_CAPABILITY_ENVELOPE_ENV])
    return _FakeProcess()

  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", _SECRET)
  monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
  registry = _registry(tmp_path, resolver)
  original = asyncio.run(_start(registry))
  resumed = asyncio.run(_start(registry, resumed_from=original["run_id"]))

  assert [request.source for request in requests] == ["start", "resume"]
  resume_request = requests[1]
  assert resume_request.required_bind == _bind(run_mode="autonomous")
  resumed_record = registry._tasks[resumed["task_id"]]
  assert resumed_record.capability_bind == registry._tasks[original["task_id"]].capability_bind
  envelope = verify_autonomous_launch_envelope(
    _SECRET,
    envelopes[-1],
  )
  assert envelope.task_id == resumed["task_id"]
  assert envelope.control_run_id == resumed["run_id"]
  assert envelope.owner_user_id == "42"
  assert envelope.bind == _bind(run_mode="autonomous")


@pytest.mark.parametrize(
  ("mode", "skill", "original_pack", "original_deliver"),
  [
    ("pack", None, "  daily-risk  ", True),
    ("skill", "risk.scan", None, False),
  ],
)
def test_resume_reuses_persisted_exact_pack_and_deliver_without_call_defaults(
  monkeypatch,
  tmp_path,
  mode,
  skill,
  original_pack,
  original_deliver,
) -> None:
  async def fake_exec(*args, **kwargs):
    _ = args, kwargs
    return _FakeProcess()

  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", _SECRET)
  monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
  registry = _registry(tmp_path)
  original = asyncio.run(
    _start(
      registry,
      mode=mode,
      skill=skill,
      pack=original_pack,
      deliver=original_deliver,
    )
  )
  resumed = asyncio.run(
    _start(
      registry,
      resumed_from=original["run_id"],
      mode=mode,
      skill=skill,
    )
  )

  expected_pack = original_pack.strip() if original_pack is not None else None
  original_record = registry._tasks[original["task_id"]]
  resumed_record = registry._tasks[resumed["task_id"]]
  assert original["mode"] == resumed["mode"] == mode
  assert original["pack"] == resumed["pack"] == expected_pack
  assert original["deliver"] is resumed["deliver"] is original_deliver
  assert original_record.pack == resumed_record.pack == expected_pack
  assert original_record.deliver is resumed_record.deliver is original_deliver

  resumed_manifest = json.loads(
    (tmp_path / f"{resumed['task_id']}.task.json").read_text(encoding="utf-8")
  )
  assert resumed_manifest["pack"] == expected_pack
  assert resumed_manifest["deliver"] is original_deliver
  rehydrated_record = _registry(tmp_path)._tasks[resumed["task_id"]]
  assert rehydrated_record.pack == expected_pack
  assert rehydrated_record.deliver is original_deliver


def test_resume_refuses_callback_that_changes_persisted_bind(monkeypatch, tmp_path) -> None:
  async def resolver(request: AutonomousCapabilityBindingRequest):
    if request.source == "resume":
      return _service_binding(
        _bind(run_mode=request.run_mode, model="changed-model"),
      )
    return _binding_for_request(request)

  async def fake_exec(*args, **kwargs):
    _ = args, kwargs
    return _FakeProcess()

  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", _SECRET)
  monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
  registry = _registry(tmp_path, resolver)
  original = asyncio.run(_start(registry))

  with pytest.raises(ValueError, match="persisted exact bind"):
    asyncio.run(_start(registry, resumed_from=original["run_id"]))


def test_schedule_fires_resolve_fresh_cron_bind_and_cron_resume_reuses_it(
  monkeypatch,
  tmp_path,
) -> None:
  requests: list[AutonomousCapabilityBindingRequest] = []
  fresh_count = 0

  async def resolver(request: AutonomousCapabilityBindingRequest):
    nonlocal fresh_count
    requests.append(request)
    if request.source == "schedule":
      fresh_count += 1
      return _service_binding(
        _bind(run_mode="cron", model=f"cron-model-{fresh_count}"),
      )
    return _binding_for_request(request)

  async def fake_exec(*args, **kwargs):
    _ = args, kwargs
    return _FakeProcess()

  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", _SECRET)
  monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
  registry = _registry(tmp_path, resolver)

  first = asyncio.run(_start(registry, schedule_id="schedule-1"))
  second = asyncio.run(_start(registry, schedule_id="schedule-1"))
  resumed = asyncio.run(_start(registry, resumed_from=first["run_id"]))

  assert [request.source for request in requests] == ["schedule", "schedule", "resume"]
  assert requests[0].run_mode == requests[1].run_mode == "cron"
  assert registry._tasks[first["task_id"]].capability_bind == _bind(
    run_mode="cron",
    model="cron-model-1",
  )
  assert registry._tasks[second["task_id"]].capability_bind == _bind(
    run_mode="cron",
    model="cron-model-2",
  )
  assert requests[2].run_mode == "cron"
  assert requests[2].required_bind == registry._tasks[first["task_id"]].capability_bind
  assert (
    registry._tasks[resumed["task_id"]].capability_bind
    == registry._tasks[first["task_id"]].capability_bind
  )


def test_schedule_user_bind_uses_secret_free_cron_envelope_and_exact_resume_handle(
  monkeypatch,
  tmp_path,
) -> None:
  handle = CredentialHandle(
    handle_id="autonomous-user:cron-exact-handle",
    provider="anthropic",
    principal="user",
    tenant_id="test-product",
    actor_id="42",
  )
  user_secret = "cron-user-secret-only-for-the-anonymous-pipe"
  materialized = MaterializedCredential(
    handle=handle,
    auth_config={
      "provider": "anthropic",
      "auth_mode": "api",
      "api_key": user_secret,
    },
  )
  requests: list[AutonomousCapabilityBindingRequest] = []
  captured_envs: list[dict[str, str]] = []
  captured_stdin: list[_FakeStdin] = []

  async def resolver(
    request: AutonomousCapabilityBindingRequest,
  ) -> AutonomousCapabilityBinding:
    requests.append(request)
    return AutonomousCapabilityBinding(
      bind=request.required_bind
      or _bind(
        run_mode="cron",
        principal="user",
        credential_ref=handle.handle_id,
      ),
      materialized_credential=materialized,
    )

  async def fake_exec(*args, **kwargs):
    _ = args
    stream = _FakeStdin()
    captured_stdin.append(stream)
    captured_envs.append(dict(kwargs["env"]))
    return _FakeProcess(stdin=stream)

  async def run_case() -> tuple[dict, dict]:
    registry = _registry(tmp_path, resolver)
    original = await _start(registry, schedule_id="schedule-1")
    resumed = await _start(registry, resumed_from=original["run_id"])
    return original, resumed

  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", _SECRET)
  monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
  original, resumed = asyncio.run(run_case())

  assert [request.source for request in requests] == ["schedule", "resume"]
  assert requests[0].run_mode == requests[1].run_mode == "cron"
  assert requests[1].required_bind == _bind(
    run_mode="cron",
    principal="user",
    credential_ref=handle.handle_id,
  )
  assert len(captured_envs) == len(captured_stdin) == 2

  for payload, env, stream in zip(
    (original, resumed),
    captured_envs,
    captured_stdin,
    strict=True,
  ):
    envelope_json = env[AUTONOMOUS_CAPABILITY_ENVELOPE_ENV]
    envelope = verify_autonomous_launch_envelope(
      _SECRET,
      envelope_json,
    )
    assert envelope.task_id == payload["task_id"]
    assert envelope.control_run_id == payload["run_id"]
    assert envelope.owner_user_id == "42"
    assert envelope.bind.run_mode == "cron"
    assert envelope.bind.credential_principal == "user"
    assert envelope.bind.credential_ref == handle.handle_id
    assert (
      env[AUTONOMOUS_CREDENTIAL_HANDOFF_ENV]
      == AUTONOMOUS_CREDENTIAL_HANDOFF_STDIN
    )
    manifest_text = (
      tmp_path / f"{payload['task_id']}.task.json"
    ).read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["capability_bind"] == envelope.bind.receipt()
    assert user_secret not in envelope_json
    assert user_secret not in json.dumps(env)
    assert user_secret not in manifest_text
    assert stream.drain_calls == 1
    assert stream.closed is True
    assert stream.wait_closed_calls == 1

    decoded = read_autonomous_credential_handoff(
      expected_handle_id=handle.handle_id,
      expected_provider=envelope.bind.provider,
      expected_principal=envelope.bind.credential_principal,
      expected_tenant_id=handle.tenant_id,
      expected_actor_id=envelope.owner_user_id,
      stream=io.BytesIO(bytes(stream.buffer)),
    )
    assert decoded.handle == handle
    assert decoded.auth_config == materialized.auth_config

  with pytest.raises(ValueError, match="does not match the signed launch"):
    read_autonomous_credential_handoff(
      expected_handle_id="autonomous-user:wrong-handle",
      expected_provider="anthropic",
      expected_principal="user",
      expected_tenant_id=handle.tenant_id,
      expected_actor_id="42",
      stream=io.BytesIO(bytes(captured_stdin[0].buffer)),
    )


def test_user_resume_refuses_changed_credential_handle_before_spawn(
  monkeypatch,
  tmp_path,
) -> None:
  original_handle = CredentialHandle(
    handle_id="autonomous-user:original-handle",
    provider="anthropic",
    principal="user",
    tenant_id="test-product",
    actor_id="42",
  )
  changed_handle = CredentialHandle(
    handle_id="autonomous-user:changed-handle",
    provider="anthropic",
    principal="user",
    tenant_id="test-product",
    actor_id="42",
  )
  spawn_calls = 0

  def materialized(handle: CredentialHandle) -> MaterializedCredential:
    return MaterializedCredential(
      handle=handle,
      auth_config={
        "provider": "anthropic",
        "auth_mode": "api",
        "api_key": f"secret-for-{handle.handle_id}",
      },
    )

  async def resolver(
    request: AutonomousCapabilityBindingRequest,
  ) -> AutonomousCapabilityBinding:
    handle = (
      changed_handle
      if request.source == "resume"
      else original_handle
    )
    return AutonomousCapabilityBinding(
      bind=request.required_bind
      or _bind(
        run_mode="autonomous",
        principal="user",
        credential_ref=original_handle.handle_id,
      ),
      materialized_credential=materialized(handle),
    )

  async def fake_exec(*args, **kwargs):
    nonlocal spawn_calls
    _ = args, kwargs
    spawn_calls += 1
    return _FakeProcess(stdin=_FakeStdin())

  async def run_case() -> None:
    registry = _registry(tmp_path, resolver)
    original = await _start(registry)
    with pytest.raises(ValueError, match="credential handle does not match"):
      await _start(registry, resumed_from=original["run_id"])

  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", _SECRET)
  monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
  asyncio.run(run_case())

  assert spawn_calls == 1


def test_resolver_helper_rejects_missing_and_mismatched_results() -> None:
  request = AutonomousCapabilityBindingRequest(
    task_id="bg_1",
    control_run_id="run-1",
    owner_user_id="42",
    raw_user_id="42",
    user_email=None,
    profile="analyst",
    mode="task",
    skill=None,
    channel="web",
    source="start",
    run_mode="autonomous",
  )

  with pytest.raises(RuntimeError, match="binding resolver is required"):
    asyncio.run(resolve_autonomous_capability_binding(None, request))
  with pytest.raises(TypeError, match="must return AutonomousCapabilityBinding"):
    asyncio.run(resolve_autonomous_capability_binding(lambda _request: object(), request))
