from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator


class ThinkingLevel(str, Enum):
  NONE = "none"
  MINIMAL = "minimal"
  LOW = "low"
  MEDIUM = "medium"
  HIGH = "high"
  MAX = "max"


@dataclass
class ModelInfo:
  id: str
  provider: str
  context_window: int = 200_000
  max_output_tokens: int = 16_384
  supports_thinking: bool = False
  supports_vision: bool = True
  supports_tool_use: bool = True
  input_cost_per_mtok: float = 0.0
  output_cost_per_mtok: float = 0.0
  cache_read_cost_per_mtok: float = 0.0
  cache_write_cost_per_mtok: float = 0.0
  compat: dict[str, Any] | None = None


@dataclass
class StreamEvent:
  type: str
  text: Any = ""
  tool_id: str = ""
  tool_name: str = ""
  tool_input_json: str = ""
  tool_input: dict[str, Any] | None = None
  thinking_text: str = ""
  signature: str = ""
  stop_reason: str = ""
  input_tokens: int = 0
  output_tokens: int = 0
  cache_read_tokens: int = 0
  cache_creation_tokens: int = 0
  raw_block: Any = None
  caller: dict[str, Any] | None = None


@dataclass
class CostEstimate:
  input_cost: float = 0.0
  output_cost: float = 0.0
  cache_read_cost: float = 0.0
  cache_write_cost: float = 0.0
  total: float = 0.0


class ModelProvider:
  name = "provider"

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    raise NotImplementedError

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    raise NotImplementedError

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    raise NotImplementedError

  def get_model_info(self, model: str) -> ModelInfo:
    raise NotImplementedError

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None,
    tools: list[dict[str, Any]],
    max_tokens: int,
    thinking_level: ThinkingLevel = ThinkingLevel.HIGH,
    **kwargs: Any,
  ) -> dict[str, Any]:
    raise NotImplementedError

  def normalize_messages(self, messages: list[dict[str, Any]], model_info: ModelInfo) -> list[dict[str, Any]]:
    return list(messages)

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncIterator[StreamEvent]:
    raise NotImplementedError

  def is_retryable_error(self, exc: Exception) -> bool:
    return False

  def estimate_cost(
    self,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
  ) -> CostEstimate:
    info = self.get_model_info(model)
    input_cost = input_tokens * info.input_cost_per_mtok / 1_000_000
    output_cost = output_tokens * info.output_cost_per_mtok / 1_000_000
    cache_read_cost = cache_read_tokens * info.cache_read_cost_per_mtok / 1_000_000
    cache_write_cost = cache_creation_tokens * info.cache_write_cost_per_mtok / 1_000_000
    return CostEstimate(
      input_cost=input_cost,
      output_cost=output_cost,
      cache_read_cost=cache_read_cost,
      cache_write_cost=cache_write_cost,
      total=input_cost + output_cost + cache_read_cost + cache_write_cost,
    )
