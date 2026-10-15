"""`hank.agent` / `hank.wait`: registration-generated bridge ops.

`agent` lowers to the turn's `run_agent` route through the bound dispatcher,
so the direct door's admission decides it; `wait` is `jobs.wait`, a read of the
executing runner's task registry. Neither is a tool: the cell learns them from
the `hello` reply, not from a copy of the catalog.
"""
from __future__ import annotations

import asyncio
import json
import sys
import textwrap
import types
from pathlib import Path
from typing import Any, AsyncIterator

from agent_gateway import AgentRunner, SessionStore, ToolDispatcher
from agent_gateway.code_execution import CodeExecutionConfig, build_code_execution
from agent_gateway.code_execution._frame_reader import BRIDGE_OPERATIONS
from agent_gateway.capability_execution import (
  ModelCallOperation,
  execute_leaf_model_call,
  list_model_call_operations,
  lookup_model_call_operation,
  register_model_call_operation,
  reset_model_call_operations_for_tests,
)
from agent_gateway.code_execution._tool_defs import make_code_execute_tool_def
from agent_gateway.event_log import EventLog
from agent_gateway.providers import ModelInfo, ModelProvider, StreamEvent
from agent_gateway.task_registry import TaskRegistry, TaskState
from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND
sys.path.insert(0, str(Path(__file__).parent))
from test_code_execution_bridge import _CODE_CALLABLE, OUTER_ID, _CaptureMcp, _tool_definitions  # noqa: E402

RUN_AGENT_DEFINITION = {
  "name": "run_agent",
  "description": "Delegate.",
  "input_schema": {"type": "object", "additionalProperties": True},
}


def _setup(run_agent: Any, *, grant_run_agent: bool) -> tuple[Any, ToolDispatcher]:
  session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
  bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))
  mcp = _CaptureMcp(frame_reader=bundle.frame_reader)
  definitions = list(_tool_definitions(bundle))
  handlers = dict(bundle.handlers)
  if grant_run_agent:
    definitions.append(RUN_AGENT_DEFINITION)
  handlers["run_agent"] = run_agent
  dispatcher = ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    mcp_client=mcp,
    local_tool_handlers=handlers,
    role="owner",
    needs_approval=lambda _name, _tool_input, _qualifier: False,
    event_log=EventLog(),
    approval_key_qualifier=bundle.approval_qualifier,
    get_tool_definitions=lambda: definitions,
  )
  bundle.bind_dispatcher(
    dispatcher,
    tool_definitions=dispatcher.get_tool_definitions,
    code_callable=_CODE_CALLABLE,
  )
  return bundle, dispatcher


class _Job:
  """A registry entry whose asyncio task settles after `delay_s`."""

  def __init__(self, task_id: str, delay_s: float) -> None:
    self.task_id = task_id
    self.state = TaskState.RUNNING
    self.completed = False
    # A settled result large enough to not fit one bridge frame: the wait row
    # must carry its identity, never this payload.
    self.result: dict[str, Any] | None = None
    self.error: dict[str, Any] | None = None
    self.task_result = types.SimpleNamespace(task_result_id=f"tr_{task_id}")

    async def _settle() -> None:
      await asyncio.sleep(delay_s)
      self.completed = True
      self.state = TaskState.COMPLETED
      self.result = {"answer": task_id.upper(), "bulk": "x" * (3 * 1024 * 1024)}

    self.asyncio_task = asyncio.get_event_loop().create_task(_settle())


async def _run_cell(dispatcher: ToolDispatcher, code: str) -> dict[str, Any]:
  result, error = await dispatcher.dispatch(
    OUTER_ID, "code_execute", {"code": textwrap.dedent(code), "host": "subprocess"},
  )
  assert error is None, error
  assert isinstance(result, dict)
  return result


def test_agent_submits_through_run_agent_and_wait_reads_the_registry() -> None:
  seen: list[dict[str, Any]] = []
  jobs: dict[str, _Job] = {}

  async def run_agent(tool_input: dict[str, Any], **_kwargs: Any) -> tuple[dict[str, Any], None]:
    seen.append(dict(tool_input))
    task_id = f"bg_{len(seen)}"
    jobs[task_id] = _Job(task_id, 0.2)
    return {"task_id": task_id, "status": "running"}, None

  async def run() -> dict[str, Any]:
    bundle, dispatcher = _setup(run_agent, grant_run_agent=True)
    runner = types.SimpleNamespace(_task_registry=types.SimpleNamespace(get=jobs.get))
    bundle.frame_reader.bind_job_runner(lambda: runner)
    return await _run_cell(dispatcher, """
      import json
      a = hank.agent("say DONE-1", role="fast")
      b = hank.agent(objective="say DONE-2", handles=["h1"])
      found = [row["name"] for row in hank.search("") if row.get("route_kind") == "bridge"]
      print(json.dumps({"a": a, "b": b, "ops": sorted(hank._operations), "found": found,
                        "described": hank.describe("wait")["arguments"]}))
      print(json.dumps(hank.wait([a["task_id"], b["task_id"], "nope"], timeout=5)))
      try:
        hank.agent("x", bogus=1)
      except TypeError as exc:
        print("TypeError", "bogus" in str(exc))
      try:
        hank.frobnicate
      except AttributeError:
        print("AttributeError")
    """)

  result = asyncio.run(run())
  submitted, waited, type_error, attribute_error = [line for line in result["stdout"].splitlines() if line.strip()]
  submitted = json.loads(submitted)
  assert submitted["ops"] == sorted(row["name"] for row in BRIDGE_OPERATIONS)
  assert submitted["a"] == {"task_id": "bg_1", "status": "running"}
  assert submitted["found"] == []  # search stays the grant ∩ code-callable listing
  assert submitted["described"] == ["task_ids", "timeout"]
  # The door's input, background forced; nothing the cell did not pass.
  assert seen == [
    {"objective": "say DONE-1", "background": True, "role": "fast"},
    {"objective": "say DONE-2", "background": True, "handles": ["h1"]},
  ]
  waited = json.loads(waited)
  assert waited["unknown"] == ["nope"]
  assert waited["pending"] == []
  assert waited["jobs"]["bg_2"] == {
    "state": "completed", "completed": True, "error": None, "task_result_id": "tr_bg_2",
  }
  assert type_error == "TypeError True"
  assert attribute_error == "AttributeError"
  # Both submissions sit on the cell's ledger as the door they went through.
  assert [entry["operation"] for entry in result["nested_calls"]] == ["run_agent", "run_agent"]


def test_wait_times_out_with_the_job_still_pending() -> None:
  jobs: dict[str, _Job] = {}

  async def run_agent(tool_input: dict[str, Any], **_kwargs: Any) -> tuple[dict[str, Any], None]:
    jobs["bg_1"] = _Job("bg_1", 30.0)
    return {"task_id": "bg_1", "status": "running"}, None

  async def run() -> dict[str, Any]:
    bundle, dispatcher = _setup(run_agent, grant_run_agent=True)
    runner = types.SimpleNamespace(_task_registry=types.SimpleNamespace(get=jobs.get))
    bundle.frame_reader.bind_job_runner(lambda: runner)
    result = await _run_cell(dispatcher, """
      import json
      a = hank.agent("slow")
      print(json.dumps(hank.wait([a["task_id"]], timeout=0.1)))
    """)
    # The observer gave up; the job did not.
    assert not jobs["bg_1"].asyncio_task.done()
    jobs["bg_1"].asyncio_task.cancel()
    return result

  result = asyncio.run(run())
  waited = json.loads(result["stdout"].strip())
  assert waited["pending"] == ["bg_1"]
  assert waited["jobs"]["bg_1"] == {"state": "running", "completed": False}


def test_agent_not_granted_and_wait_without_a_registry_refuse() -> None:
  async def run_agent(tool_input: dict[str, Any], **_kwargs: Any) -> tuple[dict[str, Any], None]:
    raise AssertionError("run_agent must not be reached when it is not granted")

  async def run() -> dict[str, Any]:
    _bundle, dispatcher = _setup(run_agent, grant_run_agent=False)
    return await _run_cell(dispatcher, """
      for call in (lambda: hank.agent("x"), lambda: hank.wait(["bg_1"])):
        try:
          call()
        except Exception as exc:
          print(exc.code)
    """)

  result = asyncio.run(run())
  assert result["stdout"].split() == ["not_granted", "bridge_unavailable"]
  assert [entry["error_code"] for entry in result["nested_calls"]] == ["not_granted"]


class _TestModelProvider(ModelProvider):
  name = "anthropic"

  def __init__(self, answer: str = "Test model answer.") -> None:
    self.answer = answer
    self.prompts_seen: list[str] = []
    self.sleep_s: float = 0.0

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name, max_output_tokens=4096, supports_thinking=False)

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    return "client"

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    pass

  def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
    return kwargs

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncIterator[StreamEvent]:
    messages = params.get("messages") or []
    if messages:
      self.prompts_seen.append(str(messages[0].get("content") or ""))
    yield StreamEvent(type="message_start", input_tokens=11)
    if self.sleep_s > 0:
      await asyncio.sleep(self.sleep_s)
    yield StreamEvent(type="text_delta", text=self.answer)
    yield StreamEvent(type="usage_update", output_tokens=7)
    yield StreamEvent(type="message_end", stop_reason="end_turn")

def _test_capability_execution(provider: Any) -> Any:
  from dataclasses import replace
  from gateway_test_support.capability_execution_test_support import (
    stub_capability_execution_resolver,
  )

  resolver = stub_capability_execution_resolver(
    default_provider="anthropic",
    default_model="claude-sonnet-4-6",
  )
  resolver = replace(resolver, adapter_resolver=lambda _adapter: provider)
  return resolver.resolve("edgar.general_completion")


def _real_runner(execution: Any, *, commercial_usage_producer: Any = None) -> AgentRunner:
  event_log = EventLog()
  return AgentRunner(
    event_log=event_log,
    dispatcher=ToolDispatcher(
      capability_bind=TEST_CAPABILITY_BIND,
      mcp_client=_CaptureMcp(),
      local_tool_handlers={},
      event_log=event_log,
      session_id="sess-bridge-ops",
    ),
    session_id="sess-bridge-ops",
    capability_execution=execution,
    user_id="alice",
    commercial_usage_producer=commercial_usage_producer,
    billing_mode="byok",
    rate_table_version="unknown",
    channel="cli",
  )

def test_completion_and_judge_bridge_ops_execute_registered_operations() -> None:
  provider = _TestModelProvider(answer='{"verdict": "supported", "findings": ["claim verified"]}')
  execution = _test_capability_execution(provider)
  runner = _real_runner(execution)

  async def run_agent(tool_input: dict[str, Any], **_kwargs: Any) -> tuple[dict[str, Any], None]:
    return {"task_id": "bg_agent", "status": "running"}, None

  async def run() -> dict[str, Any]:
    bundle, dispatcher = _setup(run_agent, grant_run_agent=True)
    bundle.frame_reader.bind_job_runner(lambda: runner)
    return await _run_cell(dispatcher, """
      import json
      c = hank.completion("general_completion", material="summarize this text")
      j = hank.judge("citation_review", state={"claim": "rev up 10%"}, questions={"supported": True})
      wait_res = hank.wait([c["task_id"], j["task_id"]], timeout=5)
      print(json.dumps({"c": c, "j": j, "wait": wait_res}))
    """)

  result = asyncio.run(run())
  output = json.loads(result["stdout"].strip())
  c = output["c"]
  j = output["j"]
  waited = output["wait"]

  assert c["status"] == "running"
  assert c["operation"] == "general_completion"
  assert c["task_id"].startswith("mc_")

  assert j["status"] == "running"
  assert j["operation"] == "citation_review"
  assert j["task_id"].startswith("mc_")

  assert waited["jobs"][c["task_id"]]["completed"] is True
  assert waited["jobs"][c["task_id"]]["state"] == "completed"

  assert waited["jobs"][j["task_id"]]["completed"] is True
  assert waited["jobs"][j["task_id"]]["state"] == "completed"

  entry_c = runner._task_registry.get(c["task_id"])
  entry_j = runner._task_registry.get(j["task_id"])
  assert entry_c is not None and entry_c.result is not None
  assert entry_j is not None and entry_j.result is not None
  assert "verdict" in entry_j.result["content"]


def test_completion_and_judge_refuse_unknown_operations_and_raw_slot_ids() -> None:
  provider = _TestModelProvider()
  execution = _test_capability_execution(provider)
  runner = _real_runner(execution)

  async def run_agent(tool_input: dict[str, Any], **_kwargs: Any) -> tuple[dict[str, Any], None]:
    return {"task_id": "bg_agent", "status": "running"}, None

  async def run() -> dict[str, Any]:
    bundle, dispatcher = _setup(run_agent, grant_run_agent=True)
    bundle.frame_reader.bind_job_runner(lambda: runner)
    return await _run_cell(dispatcher, """
      import json
      refusals = []
      for fn in [
        lambda: hank.completion("citation.review", material="test"),
        lambda: hank.completion("non_existent_op", material="test"),
        lambda: hank.judge("general_completion", state="test", questions={}),
        lambda: hank.judge("edgar.general_completion", state="test", questions={}),
      ]:
        try:
          fn()
        except Exception as exc:
          refusals.append(getattr(exc, "code", str(type(exc).__name__)))
      print(json.dumps(refusals))
    """)

  result = asyncio.run(run())
  refusals = json.loads(result["stdout"].strip())
  # All four calls are refused with unknown_operation: neither raw slot IDs nor wrong kinds are admitted.
  assert refusals == ["unknown_operation", "unknown_operation", "unknown_operation", "unknown_operation"]


def test_direct_host_door_and_kernel_door_produce_identical_bind_provenance_and_usage() -> None:
  provider = _TestModelProvider(answer="Equivalence test answer.")
  execution = _test_capability_execution(provider)
  runner = _real_runner(execution)

  # Direct host door: no runner, the resolver alone.
  direct_res = asyncio.run(execute_leaf_model_call(
    "general_completion",
    material="equivalence test material",
    resolver=runner.capability_execution.resolver,
  ))

  # Kernel door execution via cell
  async def run_agent(tool_input: dict[str, Any], **_kwargs: Any) -> tuple[dict[str, Any], None]:
    return {"task_id": "bg_agent", "status": "running"}, None

  async def run() -> dict[str, Any]:
    bundle, dispatcher = _setup(run_agent, grant_run_agent=True)
    bundle.frame_reader.bind_job_runner(lambda: runner)
    return await _run_cell(dispatcher, """
      import json
      c = hank.completion("general_completion", material="equivalence test material")
      w = hank.wait([c["task_id"]], timeout=5)
      print(json.dumps({"c": c, "w": w}))
    """)

  cell_res = asyncio.run(run())
  cell_out = json.loads(cell_res["stdout"].strip())
  cell_task_id = cell_out["c"]["task_id"]
  cell_entry = runner._task_registry.get(cell_task_id)
  assert cell_entry is not None
  assert cell_entry.result is not None

  # Identity check: same capability_id, provider, model and bind structure
  assert direct_res["bind"]["capability_id"] == cell_entry.result["capability_id"] == "edgar.general_completion"
  assert direct_res["bind"]["provider"] == cell_entry.result["provider"] == "anthropic"
  assert direct_res["bind"]["model_key"] == cell_entry.result["model"]
  assert direct_res["selection_provenance"] == cell_entry.result["selection_provenance"]
  assert direct_res["usage"] == cell_entry.result["usage"]
  assert cell_entry.result["usage"]["input_tokens"] == 11
  assert cell_entry.result["usage"]["output_tokens"] == 7


def test_model_call_cancellation_settles_as_killed() -> None:
  provider = _TestModelProvider()
  provider.sleep_s = 10.0  # Slow model call so we can cancel it
  execution = _test_capability_execution(provider)
  runner = _real_runner(execution)

  async def run_agent(tool_input: dict[str, Any], **_kwargs: Any) -> tuple[dict[str, Any], None]:
    return {"task_id": "bg_agent", "status": "running"}, None

  async def run() -> dict[str, Any]:
    bundle, dispatcher = _setup(run_agent, grant_run_agent=True)
    bundle.frame_reader.bind_job_runner(lambda: runner)
    # Start the model call in a cell, then cancel it via the registry
    task_id_holder: list[str] = []

    async def _cancel_after_started() -> None:
      for _ in range(200):
        running = runner._task_registry.list_tasks(state=TaskState.RUNNING)
        if running:
          tid = running[0].task_id
          task_id_holder.append(tid)
          runner._task_registry.kill(tid, termination_intent="cancelled")
          return
        await asyncio.sleep(0.01)

    cancel_task = asyncio.create_task(_cancel_after_started())
    cell_res = await _run_cell(dispatcher, """
      import json, time
      c = hank.completion("general_completion", material="slow cancel test")
      # Wait for a bit so the cancellation lands, then wait
      time.sleep(0.2)
      w = hank.wait([c["task_id"]], timeout=5)
      print(json.dumps({"c": c, "w": w}))
    """)
    await cancel_task
    return cell_res

  result = asyncio.run(run())
  output = json.loads(result["stdout"].strip())
  tid = output["c"]["task_id"]
  waited = output["w"]["jobs"][tid]
  assert waited["completed"] is True
  assert waited["state"] == "killed"


def _commercial_producer(sink: Any) -> Any:
  from uuid import uuid4
  from hank_contracts.commercial_claims import VerifiedCommercialClaim
  from hank_contracts.usage import CommercialUsageLineage
  from agent_gateway.commercial_usage import CommercialUsageProducer

  now = 1_780_000_000
  claim = VerifiedCommercialClaim(
    schema_version=1,
    key_id="commercial-signing-v1",
    subject="user:123",
    environment="prod",
    surface="hp1",
    commercial_account_id=uuid4(),
    agreement_id=uuid4(),
    agreement_terms_revision=2,
    offer_code="hp1_pro",
    effective_scopes=("read",),
    entitlement_revision=42,
    payer_policy_version="hp1_customer_host@v1",
    budget_policy_version="hp1_pro_budget@v1",
    shadow_rate_version="commercial_rates@2026-09-01",
    manifest_version="mcp_exposure@2026-09-01",
    authorized_work_start_deadline=now + 300,
    usage_accept_until=now + 3600,
    issued_at=now,
    expires_at=now + 300,
    context_id=uuid4(),
  )
  lineage = CommercialUsageLineage(
    source_product="hank-agent-gateway",
    workflow_run_id="wf_001",
    reservation_id="res_001",
    funding_route_id="fund_001",
    operation="messages.create",
    capability_id="edgar.general_completion",
  )
  return CommercialUsageProducer(enabled=True, claim=claim, lineage=lineage, sink=sink)


def test_leaf_call_usage_reaches_the_commercial_producer_with_its_real_state() -> None:
  """Billing end to end: cell -> leaf P4 job -> send_prompt -> CommercialUsageProducer -> sink.

  A completed call bills `succeeded`; a call cancelled after its input tokens
  streamed bills `canceled`. No forwarding seam relabels either.
  """
  billed: list[dict[str, Any]] = []

  def sink(events: list[dict[str, Any]]) -> str:
    billed.extend(events)
    return "outbox"

  provider = _TestModelProvider(answer='{"verdict": "supported"}')
  execution = _test_capability_execution(provider)
  runner = _real_runner(execution, commercial_usage_producer=_commercial_producer(sink))

  async def run_agent(tool_input: dict[str, Any], **_kwargs: Any) -> tuple[dict[str, Any], None]:
    return {"task_id": "bg_agent", "status": "running"}, None

  async def run() -> tuple[dict[str, Any], dict[str, Any]]:
    bundle, dispatcher = _setup(run_agent, grant_run_agent=True)
    bundle.frame_reader.bind_job_runner(lambda: runner)
    done = await _run_cell(dispatcher, """
      import json
      c = hank.completion("general_completion", material="billed completion")
      print(json.dumps(hank.wait([c["task_id"]], timeout=5)["jobs"][c["task_id"]]))
    """)

    provider.sleep_s = 10.0

    async def _cancel_after_tokens() -> None:
      for _ in range(500):
        running = runner._task_registry.list_tasks(state=TaskState.RUNNING)
        if running and provider.prompts_seen[-1:] == ["billed cancel"]:
          await asyncio.sleep(0.05)  # message_start has streamed; the sleep holds the call open
          runner._task_registry.kill(running[0].task_id, termination_intent="cancelled")
          return
        await asyncio.sleep(0.01)

    cancel = asyncio.create_task(_cancel_after_tokens())
    killed = await _run_cell(dispatcher, """
      import json, time
      c = hank.completion("general_completion", material="billed cancel")
      time.sleep(0.3)
      print(json.dumps(hank.wait([c["task_id"]], timeout=5)["jobs"][c["task_id"]]))
    """)
    await cancel
    return json.loads(done["stdout"].strip()), json.loads(killed["stdout"].strip())

  done, killed = asyncio.run(run())
  assert done["state"] == "completed"
  assert killed["state"] == "killed"

  states = [(p["usage_state"], p["uncached_input_tokens"], p["billable_output_tokens"]) for p in billed]
  assert states == [("succeeded", 11, 7), ("canceled", 11, 0)]
  assert {p["capability_bind"]["capability_id"] for p in billed} == {"edgar.general_completion"}


def test_prompt_names_registered_model_call_operations() -> None:
  tool_def = make_code_execute_tool_def(CodeExecutionConfig(), available_hosts=("auto",))
  desc = tool_def["description"]
  assert "general_completion" in desc
  assert "citation_review" in desc
  assert "verification" in desc
  assert "hank.completion(operation, material, role, effort, evidence_handles)" in desc
  assert "hank.judge(operation, state, questions, role, effort, evidence_handles)" in desc
  assert "hank.workpool(items, operation, mode, role, effort, context_handles, timeout)" in desc
  assert "hank.judge_batch(operation, states, questions, role, effort, evidence_handles, wait, timeout)" in desc


def test_describe_model_call_and_bridge_operations() -> None:
  provider = _TestModelProvider()
  execution = _test_capability_execution(provider)
  runner = _real_runner(execution)

  async def run_agent(tool_input: dict[str, Any], **_kwargs: Any) -> tuple[dict[str, Any], None]:
    return {"task_id": "bg_agent", "status": "running"}, None

  async def run() -> dict[str, Any]:
    bundle, dispatcher = _setup(run_agent, grant_run_agent=True)
    bundle.frame_reader.bind_job_runner(lambda: runner)
    return await _run_cell(dispatcher, """
      import json
      d_comp = hank.describe("completion")
      d_judge = hank.describe("judge")
      d_gen = hank.describe("general_completion")
      d_cit = hank.describe("citation_review")
      print(json.dumps({
        "comp": d_comp,
        "judge": d_judge,
        "gen": d_gen,
        "cit": d_cit,
      }))
    """)

  result = asyncio.run(run())
  output = json.loads(result["stdout"].strip())
  assert output["comp"]["route_kind"] == "bridge"
  assert output["comp"]["name"] == "completion"

  assert output["judge"]["route_kind"] == "bridge"
  assert output["judge"]["name"] == "judge"

  assert output["gen"]["route_kind"] == "model_call"
  assert output["gen"]["name"] == "general_completion"
  assert output["gen"]["execution_class"] == "edgar.general_completion"

  assert output["cit"]["route_kind"] == "model_call"
  assert output["cit"]["name"] == "citation_review"
  assert output["cit"]["kind"] == "judgment"


def test_workpool_submits_controller_job_and_child_jobs() -> None:
  provider = _TestModelProvider()
  execution = _test_capability_execution(provider)
  runner = _real_runner(execution)

  child_run_count = 0

  async def run_agent(tool_input: dict[str, Any], **_kwargs: Any) -> tuple[dict[str, Any], None]:
    nonlocal child_run_count
    child_run_count += 1
    tid = f"bg_agent_{child_run_count}"
    entry = runner._task_registry.register(task_type="agent", task_id=tid)
    entry.state = TaskState.COMPLETED
    entry.result = {"content": f"Analysis for {tool_input.get('objective')}"}
    return {"task_id": tid, "status": "running"}, None

  async def run() -> dict[str, Any]:
    bundle, dispatcher = _setup(run_agent, grant_run_agent=True)
    bundle.frame_reader.bind_job_runner(lambda: runner)
    return await _run_cell(dispatcher, """
      import json
      items = [
        {"key": "item_a", "objective": "analyze the first item"},
        {"key": "item_b", "objective": "analyze the second item"},
      ]
      wp = hank.workpool(items=items)
      w = hank.wait([wp["task_id"]], timeout=5)
      print(json.dumps({"wp": wp, "wait": w}))
    """)

  result = asyncio.run(run())
  output = json.loads(result["stdout"].strip())
  wp = output["wp"]
  waited = output["wait"]

  assert wp["status"] == "running"
  assert wp["task_id"].startswith("wp_")
  assert wp["keys"] == ["item_a", "item_b"]
  assert "item_a" in wp["jobs"]
  assert "item_b" in wp["jobs"]

  controller_id = wp["task_id"]
  assert waited["jobs"][controller_id]["completed"] is True
  assert waited["jobs"][controller_id]["state"] == "completed"

  controller_entry = runner._task_registry.get(controller_id)
  assert controller_entry is not None
  assert controller_entry.state == TaskState.COMPLETED
  assert controller_entry.completed_at is not None
  assert controller_entry.notification_generation > 0
  assert controller_entry.result is not None
  assert controller_entry.result["items"]["item_a"]["status"] == "completed"
  assert controller_entry.result["items"]["item_b"]["status"] == "completed"
  assert child_run_count == 2
  wp_calls = [c for c in result["nested_calls"] if c["operation"] == "workpool"]
  assert len(wp_calls) == 1
  assert wp_calls[0]["ok"] is True
  assert wp_calls[0]["duration_ms"] >= 0


def test_judge_batch_executes_keyed_judgments_under_registered_op() -> None:
  provider = _TestModelProvider(answer='{"verdict": "supported", "findings": ["verified"]}')
  execution = _test_capability_execution(provider)
  runner = _real_runner(execution)

  async def run_agent(tool_input: dict[str, Any], **_kwargs: Any) -> tuple[dict[str, Any], None]:
    return {"task_id": "bg_agent", "status": "running"}, None

  async def run() -> dict[str, Any]:
    bundle, dispatcher = _setup(run_agent, grant_run_agent=True)
    bundle.frame_reader.bind_job_runner(lambda: runner)
    return await _run_cell(dispatcher, """
      import json
      states = {
        "claim_1": {"claim": "revenue up 10%"},
        "claim_2": {"claim": "margin expanded 200 bps"},
      }
      questions = {"supported": True}
      jb = hank.judge_batch("citation_review", states=states, questions=questions)
      print(json.dumps(jb))
    """)

  result = asyncio.run(run())
  output = json.loads(result["stdout"].strip())
  assert output["status"] == "completed"
  assert output["operation"] == "citation_review"
  assert "claim_1" in output["judgments"]
  assert "claim_2" in output["judgments"]
  assert "verdict" in output["judgments"]["claim_1"]["content"]
  assert "verdict" in output["judgments"]["claim_2"]["content"]
  jb_calls = [c for c in result["nested_calls"] if c["operation"] == "judge_batch"]
  assert len(jb_calls) == 1
  assert jb_calls[0]["ok"] is True
  assert jb_calls[0]["duration_ms"] >= 0

def test_workpool_and_judge_batch_refusals() -> None:
  provider = _TestModelProvider()
  execution = _test_capability_execution(provider)
  runner = _real_runner(execution)

  async def run_agent(tool_input: dict[str, Any], **_kwargs: Any) -> tuple[dict[str, Any], None]:
    return {"task_id": "bg_agent", "status": "running"}, None

  # 1. workpool not granted when run_agent is not granted
  async def run_no_grant() -> dict[str, Any]:
    bundle, dispatcher = _setup(run_agent, grant_run_agent=False)
    bundle.frame_reader.bind_job_runner(lambda: runner)
    return await _run_cell(dispatcher, """
      try:
        hank.workpool(items=[{"key": "k1", "objective": "test"}])
        print("ok")
      except Exception as exc:
        print("error:", getattr(exc, "code", str(exc)))
    """)

  res_no_grant = asyncio.run(run_no_grant())
  assert "not_granted" in res_no_grant["stdout"]

  # 2. judge_batch unknown operation
  async def run_unknown_op() -> dict[str, Any]:
    bundle, dispatcher = _setup(run_agent, grant_run_agent=True)
    bundle.frame_reader.bind_job_runner(lambda: runner)
    return await _run_cell(dispatcher, """
      try:
        hank.judge_batch("nonexistent_judge", states={"k1": {}}, questions={})
        print("ok")
      except Exception as exc:
        print("error:", getattr(exc, "code", str(exc)))
    """)

  res_unknown_op = asyncio.run(run_unknown_op())
  assert "unknown_operation" in res_unknown_op["stdout"]

  # 3. workpool duplicate item keys
  async def run_dup_keys() -> dict[str, Any]:
    bundle, dispatcher = _setup(run_agent, grant_run_agent=True)
    bundle.frame_reader.bind_job_runner(lambda: runner)
    return await _run_cell(dispatcher, """
      try:
        hank.workpool(items=[{"key": "dup", "objective": "1"}, {"key": "dup", "objective": "2"}])
        print("ok")
      except Exception as exc:
        print("error:", getattr(exc, "code", str(exc)))
    """)

  res_dup = asyncio.run(run_dup_keys())
  assert "protocol_error" in res_dup["stdout"]


def test_workpool_timeout_leaves_controller_interrupted() -> None:
  provider = _TestModelProvider()
  execution = _test_capability_execution(provider)
  runner = _real_runner(execution)

  # Child stays RUNNING indefinitely
  async def run_agent(tool_input: dict[str, Any], **_kwargs: Any) -> tuple[dict[str, Any], None]:
    tid = "bg_slow_agent"
    entry = runner._task_registry.register(task_type="agent", task_id=tid)
    entry.state = TaskState.RUNNING
    return {"task_id": tid, "status": "running"}, None

  async def run() -> dict[str, Any]:
    bundle, dispatcher = _setup(run_agent, grant_run_agent=True)
    bundle.frame_reader.bind_job_runner(lambda: runner)
    return await _run_cell(dispatcher, """
      import json, time
      wp = hank.workpool(items=[{"key": "slow", "objective": "slow task"}], timeout=0.1)
      time.sleep(0.3)
      print(json.dumps(wp))
    """)

  result = asyncio.run(run())
  output = json.loads(result["stdout"].strip())
  controller_id = output["task_id"]

  controller_entry = runner._task_registry.get(controller_id)
  assert controller_entry is not None
  assert controller_entry.state == TaskState.INTERRUPTED
  assert controller_entry.result is not None
  assert controller_entry.result["status"] == "timed_out"
  assert controller_entry.result["items"]["slow"]["status"] == "running"


def test_workpool_controller_continues_dispatch_after_cell_closure() -> None:
  provider = _TestModelProvider()
  execution = _test_capability_execution(provider)
  runner = _real_runner(execution)

  dispatch_started: list[str] = []
  dispatch_finished: list[str] = []
  barrier = asyncio.Event()

  async def run_agent(tool_input: dict[str, Any], **_kwargs: Any) -> tuple[dict[str, Any], None]:
    obj = str(tool_input.get("objective") or "")
    dispatch_started.append(obj)
    if obj == "task_2":
      await barrier.wait()
    dispatch_finished.append(obj)
    tid = f"bg_{obj}"
    entry = runner._task_registry.register(task_type="agent", task_id=tid)
    entry.state = TaskState.COMPLETED
    entry.result = {"content": f"Done {obj}"}
    return {"task_id": tid, "status": "running"}, None

  async def run_test() -> None:
    bundle, dispatcher = _setup(run_agent, grant_run_agent=True)
    bundle.frame_reader.bind_job_runner(lambda: runner)

    cell_task = asyncio.create_task(
      _run_cell(
        dispatcher,
        """
        import json
        wp = hank.workpool(items=[
          {"key": "k1", "objective": "task_1"},
          {"key": "k2", "objective": "task_2"},
          {"key": "k3", "objective": "task_3"},
        ])
        print(json.dumps(wp))
        """,
      )
    )

    while "task_2" not in dispatch_started:
      await asyncio.sleep(0.005)

    assert "task_3" not in dispatch_started

    # Cancel cell and close frame reader to simulate cell loss mid-dispatch
    cell_task.cancel()
    try:
      await cell_task
    except (asyncio.CancelledError, Exception):
      pass
    await bundle.frame_reader.aclose()

    ctrl_tasks = [t for t in runner._task_registry._tasks.values() if t.task_type == "controller"]
    assert len(ctrl_tasks) == 1
    ctrl_entry = ctrl_tasks[0]
    assert ctrl_entry.state == TaskState.RUNNING

    # Release blocked task_2; controller must finish dispatching task_3
    barrier.set()
    if ctrl_entry.asyncio_task:
      await ctrl_entry.asyncio_task

    assert ctrl_entry.state == TaskState.COMPLETED
    assert dispatch_finished == ["task_1", "task_2", "task_3"]
    assert ctrl_entry.result is not None
    assert len(ctrl_entry.result["items"]) == 3
  asyncio.run(run_test())

def test_workpool_controller_reconstructs_from_event_log() -> None:
  import os
  provider = _TestModelProvider()
  execution = _test_capability_execution(provider)
  runner = _real_runner(execution)

  async def run_agent(tool_input: dict[str, Any], **_kwargs: Any) -> tuple[dict[str, Any], None]:
    tid = f"bg_{os.urandom(2).hex()}"
    entry = runner._task_registry.register(task_type="agent", task_id=tid)
    entry.state = TaskState.COMPLETED
    entry.result = {"content": "Done"}
    return {"task_id": tid, "status": "running"}, None

  async def run() -> dict[str, Any]:
    bundle, dispatcher = _setup(run_agent, grant_run_agent=True)
    bundle.frame_reader.bind_job_runner(lambda: runner)
    return await _run_cell(dispatcher, """
      import json
      wp = hank.workpool(items=[{"key": "k1", "objective": "test objective"}])
      w = hank.wait([wp["task_id"]], timeout=5)
      print(json.dumps({"wp": wp, "wait": w}))
    """)

  result = asyncio.run(run())
  output = json.loads(result["stdout"].strip())
  cid = output["wp"]["task_id"]

  event_log = getattr(runner, "_log", None) or getattr(runner, "_event_log", None)
  assert event_log is not None
  events = [e.event for e in getattr(event_log, "_entries", ())]
  fresh_registry = TaskRegistry()
  fresh_registry.load_from_events(events)

  reconstructed = fresh_registry.get(cid)
  assert reconstructed is not None
  assert reconstructed.state == TaskState.COMPLETED
  assert reconstructed.result is not None
  assert reconstructed.result["status"] == "completed"
  assert "k1" in reconstructed.result["items"]


def test_workpool_controller_handles_refused_item_without_timeout() -> None:
  provider = _TestModelProvider()
  execution = _test_capability_execution(provider)
  runner = _real_runner(execution)

  async def run_agent(tool_input: dict[str, Any], **_kwargs: Any) -> tuple[dict[str, Any], None]:
    obj = str(tool_input.get("objective") or "")
    if obj == "task_bad":
      raise RuntimeError("Admission refused item")
    tid = f"bg_{obj}"
    entry = runner._task_registry.register(task_type="agent", task_id=tid)
    entry.state = TaskState.COMPLETED
    entry.result = {"content": f"Done {obj}"}
    return {"task_id": tid, "status": "running"}, None

  async def run_test() -> None:
    bundle, dispatcher = _setup(run_agent, grant_run_agent=True)
    bundle.frame_reader.bind_job_runner(lambda: runner)
    cell_res = await _run_cell(dispatcher, """
      import json
      wp = hank.workpool(items=[
        {"key": "k1", "objective": "task_good"},
        {"key": "k2", "objective": "task_bad"},
      ])
      print(json.dumps(wp))
    """)
    output = json.loads(cell_res["stdout"].strip())
    cid = output["task_id"]
    assert output["jobs"]["k1"]["task_id"] == "bg_task_good"
    assert output["jobs"]["k1"]["status"] == "running"
    assert output["jobs"]["k2"]["status"] == "refused"
    assert "task_id" not in output["jobs"]["k2"]
    assert output["jobs"]["k2"]["error"]

    entry = runner._task_registry.get(cid)
    assert entry is not None
    if entry.asyncio_task:
      await entry.asyncio_task

    assert entry.state == TaskState.COMPLETED
    assert entry.result is not None
    assert entry.result["status"] == "completed"
    assert entry.result["items"]["k1"]["status"] == "completed"
    assert entry.result["items"]["k2"]["status"] == "refused"
    assert entry.result["items"]["k2"]["error"]

  asyncio.run(run_test())

