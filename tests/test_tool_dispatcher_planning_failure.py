# ruff: noqa: E402
"""An exact-write planner or a local tool that raises reaches the agent as its own error.

HUBS 2026-10-06 (``docs/inbox/planned-write-planning-catch-all-hides-the-planner-error.md``):
``fms_persist_forecast_assumption_set`` planning raised ``LineageContractError`` twice and
the agent read only "could not produce a trusted exact-write plan", no field, no next call.
HUBS run 2 (seq 46): its executor raised ``ForecastArtifactRepositoryError`` and the
runner renamed it ``internal_error``, message only; (seq 62) a stored-model
``ValidationError`` was renamed ``ENVIRONMENT_ERROR`` with no field.
"""

from __future__ import annotations
from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND

import asyncio
import sys
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, ValidationError

PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import ToolDispatcher
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.tool_dispatcher_helpers import ToolResult


class _NullMcp(McpClientManager):
  def __init__(self) -> None:
    super().__init__(config_path=None)

  def is_mcp_tool(self, name: str) -> bool:
    _ = name
    return False

  def get_server_for_tool(self, name: str) -> str | None:
    _ = name
    return None


class _RaisingPlannedHandler:
  PLANNING_IDENTITY = "change_set"

  def __init__(self, exc: Exception) -> None:
    self._exc = exc

  async def __call__(self, *_args: object, **_kwargs: object) -> ToolResult:
    raise AssertionError("a planned write never runs the legacy handler")

  async def plan_change(self, _tool_input: dict[str, Any], **_kwargs: object) -> tuple[Any, Any]:
    raise self._exc

  async def execute_prepared_change(self, *_args: object, **_kwargs: object) -> ToolResult:
    raise AssertionError("planning failed; nothing executes")


def _unused_validator(*_args: object) -> Mapping[str, Any]:
  raise AssertionError("planning failed before a plan existed")


def _dispatch(exc: Exception, tool_input: dict[str, Any]) -> ToolResult:
  dispatcher = ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    role="owner",
    mcp_client=_NullMcp(),
    local_tool_handlers={"planned_write": _RaisingPlannedHandler(exc)},
    plan_validator=_unused_validator,
  )
  return asyncio.run(dispatcher.dispatch("call-plan", "planned_write", tool_input))


class _LineageContractError(ValueError):
  pass


def test_uncoded_planner_error_carries_type_reason_and_a_next_call() -> None:
  tool_input = {"context_ref": "facx3.abc", "judgment": {"assumptions": []}}
  result, error = _dispatch(
    _LineageContractError("artifact lineage source alias 'S3' names no issued read"),
    tool_input,
  )

  assert result is None and error is not None
  assert error["code"] == "planned_write_planning_failed"
  assert "_LineageContractError" in error["message"]
  assert "artifact lineage source alias 'S3' names no issued read" in error["message"]
  data = error["data"]
  assert data["error_type"] == "_LineageContractError"
  assert data["reason"] == "artifact lineage source alias 'S3' names no issued read"
  assert data["field"] is None
  assert data["next_call"] == {"tool_name": "planned_write", "arguments": tool_input}


class _CodedPlannerError(RuntimeError):
  def __init__(self) -> None:
    super().__init__("no current ModelVersion is selected for research_file_id=1")
    self.code = "current_model_not_found"
    self.fix = "build the model first"
    self.data = {
      "field": "judgment.current_model",
      "next_call": {"tool_name": "build_model", "arguments": {"research_file_id": 1}},
    }


def test_planner_error_field_fix_code_and_next_call_pass_through() -> None:
  result, error = _dispatch(_CodedPlannerError(), {"judgment": {}})

  assert result is None and error is not None
  assert error["code"] == "planned_write_planning_failed"
  data = error["data"]
  assert data["field"] == "judgment.current_model"
  assert data["owner_code"] == "current_model_not_found"
  assert data["fix"] == "build the model first"
  assert data["next_call"] == {"tool_name": "build_model", "arguments": {"research_file_id": 1}}


class _RaisingLocalTool:
  def __init__(self, exc: Exception) -> None:
    self._exc = exc

  async def __call__(self, *_args: object, **_kwargs: object) -> ToolResult:
    raise self._exc


def _dispatch_local(exc: Exception, tool_input: dict[str, Any]) -> ToolResult:
  dispatcher = ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    role="owner",
    mcp_client=_NullMcp(),
    local_tool_handlers={"local_write": _RaisingLocalTool(exc)},
  )
  return asyncio.run(dispatcher.dispatch("call-exec", "local_write", tool_input))


class _ForecastArtifactRepositoryError(RuntimeError):
  def __init__(self) -> None:
    super().__init__("HistoricalFoundation does not close the exact historical seed denominator")
    self.code = "historical_seed_denominator_open"
    self.field = "judgment.assumptions[0].seed"
    self.next_call = {"tool_name": "get_historical_foundation_context", "arguments": {"ticker": "HUBS"}}


def test_local_tool_error_carries_type_field_code_and_next_call() -> None:
  result, error = _dispatch_local(_ForecastArtifactRepositoryError(), {"judgment": {}})

  assert result is None and error is not None
  assert error["code"] == "local_tool_failed"
  data = error["data"]
  assert data["error_type"] == "_ForecastArtifactRepositoryError"
  assert data["reason"] == "HistoricalFoundation does not close the exact historical seed denominator"
  assert data["field"] == "judgment.assumptions[0].seed"
  assert data["owner_code"] == "historical_seed_denominator_open"
  assert data["next_call"] == {
    "tool_name": "get_historical_foundation_context",
    "arguments": {"ticker": "HUBS"},
  }


class _ComparativeAnalysisRow(BaseModel):
  model_config = ConfigDict(extra="forbid", title="ComparativeAnalysisRow")

  peer_ticker: str
  metrics: list[float]


def _construction_defect() -> ValidationError:
  try:
    _ComparativeAnalysisRow.model_validate({"peer_ticker": "HUBS", "metrics": [1.0, "n/a"]})
  except ValidationError as exc:
    return exc
  raise AssertionError("the stored model must reject the row")


def test_stored_model_validation_error_names_model_and_field() -> None:
  tool_input = {"judgment": {"peers": ["HUBS"]}}
  result, error = _dispatch_local(_construction_defect(), tool_input)

  assert result is None and error is not None
  assert error["code"] == "local_tool_failed"
  data = error["data"]
  assert data["error_type"] == "ValidationError"
  assert data["model"] == "ComparativeAnalysisRow"
  assert data["field"] == "metrics.1"
  assert data["next_call"] == {"tool_name": "local_write", "arguments": tool_input}
