from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


RUN_IDENTITY_MCP_ENV = "AGENT_SKILL_RUN_ID"
RUN_IDENTITY_MCP_TOOLS_ENV = "AGENT_SKILL_RUN_ID_TOOLS"

# These are the only MCP model-building chains allowed to consume the
# server-owned run identity. The report door is gateway-local, so it receives
# the same identity through ToolExecutionContext rather than subprocess env.
MODEL_RUN_IDENTITY_MCP_TOOLS = frozenset({
  "build_model",
  "prepare_model_build",
})
MODEL_RUN_IDENTITY_LOCAL_TOOLS = frozenset({"fms_report_build_model"})
MODEL_RUN_IDENTITY_FORBIDDEN_TOOLS = frozenset({"author_model_spec"})
RUN_IDENTITY_MAX_LENGTH = 128
_RUN_IDENTITY_PATTERN = re.compile(
  rf"^[A-Za-z0-9][A-Za-z0-9._:-]{{0,{RUN_IDENTITY_MAX_LENGTH - 1}}}$"
)


class RunIdentityCarrierError(ValueError):
  """A server-owned run identity cannot be routed without ambiguity."""

  def __init__(self, code: str, message: str | None = None) -> None:
    if message is None:
      message = code
      code = "run_identity_invalid"
    super().__init__(message)
    self.code = code


def validate_run_identity(value: object) -> str:
  """Return one bounded, path-safe identity or fail closed."""

  if type(value) is not str or _RUN_IDENTITY_PATTERN.fullmatch(value) is None:
    raise RunIdentityCarrierError(
      "run_identity_invalid",
      "run identity must be a canonical ASCII token of at most 128 characters",
    )
  return value


@dataclass(frozen=True, slots=True)
class RunIdentityCarrier:
  """Validated server-owned identity shared by runtime transport carriers."""

  run_id: str

  def __post_init__(self) -> None:
    validate_run_identity(self.run_id)

  @classmethod
  def from_optional(cls, value: str | None) -> "RunIdentityCarrier | None":
    if value is None:
      return None
    return cls(value)

  def mcp_environment(self) -> dict[str, str]:
    return {
      RUN_IDENTITY_MCP_ENV: self.run_id,
      RUN_IDENTITY_MCP_TOOLS_ENV: ",".join(
        sorted(MODEL_RUN_IDENTITY_MCP_TOOLS)
      ),
    }


def model_run_identity_for_tool(
  tool_name: str,
  carrier: RunIdentityCarrier | None,
) -> str | None:
  """Resolve identity only for the run-bound model chains.

  The judgment-author chain is an explicit negative route: attempting to
  attach identity there is a construction error instead of a silent widening.
  """

  normalized_name = str(tool_name or "").strip()
  if normalized_name in MODEL_RUN_IDENTITY_FORBIDDEN_TOOLS:
    if carrier is not None:
      raise RunIdentityCarrierError(
        "run_identity_forbidden",
        f"{normalized_name} does not accept run identity"
      )
    return None
  if normalized_name in (
    MODEL_RUN_IDENTITY_MCP_TOOLS | MODEL_RUN_IDENTITY_LOCAL_TOOLS
  ):
    return carrier.run_id if carrier is not None else None
  return None


def mcp_metadata_skill_run_id(
  tool_name: str,
  skill_run_id: str | None,
) -> str | None:
  """Route lifecycle authority or legacy attribution to MCP metadata.

  Only prepare/build treat this field as authority. Other MCP tools retain the
  historical attribution value; the author chain is explicitly identity-free.
  """

  normalized_name = str(tool_name or "").strip()
  if normalized_name in MODEL_RUN_IDENTITY_FORBIDDEN_TOOLS:
    return None
  if normalized_name not in MODEL_RUN_IDENTITY_MCP_TOOLS:
    normalized_attribution = str(skill_run_id or "").strip()
    return normalized_attribution or None
  carrier = RunIdentityCarrier.from_optional(skill_run_id)
  return carrier.run_id if carrier is not None else None


def inject_run_identity_into_mcp_server_configs(
  configs: Mapping[str, Any],
  *,
  server_names: set[str] | frozenset[str],
  carrier: RunIdentityCarrier | None,
) -> dict[str, Any]:
  """Copy SDK MCP configs and bind the run carrier to selected stdio servers."""

  copied = dict(configs)
  if carrier is None:
    return copied
  carrier_env = carrier.mcp_environment()
  for server_name in sorted(set(server_names) & set(copied)):
    raw_config = copied[server_name]
    if not isinstance(raw_config, Mapping):
      raise RunIdentityCarrierError(
        "run_identity_transport_invalid",
        f"MCP server {server_name!r} cannot carry run identity"
      )
    config = dict(raw_config)
    if not str(config.get("command") or "").strip():
      raise RunIdentityCarrierError(
        "run_identity_transport_invalid",
        f"MCP server {server_name!r} requires a run-scoped stdio carrier"
      )
    raw_env = config.get("env")
    if raw_env is not None and not isinstance(raw_env, Mapping):
      raise RunIdentityCarrierError(
        "run_identity_transport_invalid",
        f"MCP server {server_name!r} has invalid environment config"
      )
    env = {
      str(key): str(value)
      for key, value in dict(raw_env or {}).items()
    }
    for key, expected in carrier_env.items():
      existing = env.get(key)
      if existing is not None and existing != expected:
        raise RunIdentityCarrierError(
          "run_identity_mismatch",
          f"MCP server {server_name!r} attempted to override {key}"
        )
      env[key] = expected
    config["env"] = env
    copied[server_name] = config
  return copied


__all__ = [
  "MODEL_RUN_IDENTITY_FORBIDDEN_TOOLS",
  "MODEL_RUN_IDENTITY_LOCAL_TOOLS",
  "MODEL_RUN_IDENTITY_MCP_TOOLS",
  "RUN_IDENTITY_MCP_ENV",
  "RUN_IDENTITY_MAX_LENGTH",
  "RUN_IDENTITY_MCP_TOOLS_ENV",
  "RunIdentityCarrier",
  "RunIdentityCarrierError",
  "inject_run_identity_into_mcp_server_configs",
  "mcp_metadata_skill_run_id",
  "model_run_identity_for_tool",
  "validate_run_identity",
]
