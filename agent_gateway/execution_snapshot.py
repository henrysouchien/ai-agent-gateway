"""Canonical construction of immutable child execution prompts and mechanics."""

from __future__ import annotations

import datetime
from typing import Any

from agent_workflow_contracts import (
  AgentExecutionSnapshot,
  AgentOperationSnapshot,
  AgentResumeMechanics,
  ResultRequirement,
)


def render_result_instructions(requirement: ResultRequirement) -> str:
  """Render the only model-facing result contract: a normal final message.

  Wire models are runtime-owned.  A child may call typed tools while working,
  but it never serializes the runtime's result envelope or submits a typed
  sidecar as its answer.
  """

  if not isinstance(requirement, ResultRequirement):
    raise TypeError("requirement must be a ResultRequirement")
  if requirement.mode != "narrative":
    raise ValueError(
      "agent execution supports terminal-message results only; structured "
      "records must be materialized by deterministic runtime code"
    )
  return "\n".join((
    "## Child Result",
    "",
    "Return the complete substantive result in your final assistant message.",
    "Your final assistant message is the authoritative child result.",
  ))


def build_agent_execution_snapshot(
  *,
  operation: AgentOperationSnapshot,
  result_instructions: str,
  admission_date: str | None = None,
  persisted_methodology_state: Any | None = None,
  methodology_state_instructions: str | None = None,
  context_instructions: str | None = None,
  max_turns: int | None,
  timeout_seconds: float | None,
  client_timeout_seconds: float,
  max_tokens: int,
  cost_observation_threshold_usd: float | None,
  max_resume_chain_depth: int,
  resume_instruction: str | None = None,
  max_budget_usd: float | None = None,
) -> AgentExecutionSnapshot:
  """Freeze every model-visible prompt byte and execution mechanic at admission."""

  if not isinstance(operation, AgentOperationSnapshot):
    raise TypeError("operation must be an AgentOperationSnapshot")
  exact_date = (
    datetime.date.today().isoformat()
    if admission_date is None
    else admission_date
  )
  sections = [operation.instructions, f"Current date: {exact_date}"]
  if persisted_methodology_state is not None:
    state_instructions = str(methodology_state_instructions or "").strip()
    if not state_instructions:
      raise ValueError(
        "persisted methodology state requires exact rendered instructions"
      )
    sections.append(state_instructions)
  elif methodology_state_instructions is not None:
    raise ValueError(
      "methodology state instructions require persisted methodology state"
    )
  if context_instructions is not None:
    exact_context = context_instructions.strip()
    if not exact_context:
      raise ValueError("context instructions must be non-empty when provided")
    sections.append(exact_context)
  exact_result_instructions = result_instructions.strip()
  if not exact_result_instructions:
    raise ValueError("result instructions must be non-empty")
  sections.append(exact_result_instructions)
  exact_resume_instruction = (
    resume_instruction.strip() if resume_instruction is not None else None
  )
  if resume_instruction is not None and not exact_resume_instruction:
    raise ValueError("resume instruction must be non-empty when provided")
  return AgentExecutionSnapshot(
    system_prompt="\n\n".join(sections),
    admission_date=exact_date,
    persisted_methodology_state=persisted_methodology_state,
    result_instructions=exact_result_instructions,
    max_turns=max_turns,
    timeout_seconds=timeout_seconds,
    client_timeout_seconds=client_timeout_seconds,
    max_tokens=max_tokens,
    cost_observation_threshold_usd=cost_observation_threshold_usd,
    max_budget_usd=max_budget_usd,
    resume_mechanics=AgentResumeMechanics(
      resumable=operation.resumable,
      max_chain_depth=(max_resume_chain_depth if operation.resumable else 0),
      transcript_strategy="durable_reconstruction",
      prompt_strategy="reuse_exact",
      tool_grant_strategy="reissue_exact",
      control_message_strategy="admitted_exact",
    ),
    resume_instruction=exact_resume_instruction,
  )


def resume_agent_execution_snapshot(
  snapshot: AgentExecutionSnapshot,
  *,
  resume_instruction: str,
) -> AgentExecutionSnapshot:
  """Bind one exact continuation instruction without rebuilding prior config."""

  if not isinstance(snapshot, AgentExecutionSnapshot):
    raise TypeError("snapshot must be an AgentExecutionSnapshot")
  exact_instruction = resume_instruction.strip()
  if not exact_instruction:
    raise ValueError("resume instruction must be non-empty")
  if not snapshot.resume_mechanics.resumable:
    raise ValueError("execution snapshot is not resumable")
  return snapshot.model_copy(update={"resume_instruction": exact_instruction})


__all__ = [
  "build_agent_execution_snapshot",
  "render_result_instructions",
  "resume_agent_execution_snapshot",
]
