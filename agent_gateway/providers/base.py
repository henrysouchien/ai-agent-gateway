from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, AsyncIterator, Literal

from ..auth import ProviderCredentialFailure
from ..thinking import EffortResolution, ThinkingLevel


_STATUS_CODE_RE = re.compile(r"\b(400|401|403|404|429|5\d\d)\b")
_BILLING_PATTERNS = (
  re.compile(r"\bbilling\b", re.IGNORECASE),
  re.compile(r"\bquota\b", re.IGNORECASE),
  re.compile(r"\binsufficient[_\s-]+quota\b", re.IGNORECASE),
  re.compile(r"\bcredit(?:s)?\b", re.IGNORECASE),
  re.compile(r"\bpayment\b", re.IGNORECASE),
  re.compile(r"\bsubscription\b", re.IGNORECASE),
  re.compile(r"\bpermission denied\b", re.IGNORECASE),
  re.compile(r"\bpermissiondenied\b", re.IGNORECASE),
)
_AUTH_PATTERNS = (
  re.compile(r"\bunauthorized\b", re.IGNORECASE),
  re.compile(r"\bauth(?:entication|orization)?(?:\s+failed|\s+error)?\b", re.IGNORECASE),
  re.compile(r"\binvalid api key\b", re.IGNORECASE),
  re.compile(r"\bexpired (?:token|credential|key)\b", re.IGNORECASE),
)
_RATE_LIMIT_PATTERNS = (
  re.compile(r"\brate(?:\s+limit(?:ed|ing)?|\s+limited)\b", re.IGNORECASE),
  re.compile(r"\btoo many requests\b", re.IGNORECASE),
)
_CONTEXT_LENGTH_PATTERNS = (
  re.compile(r"\bcontext[_\s-]*length[_\s-]*(?:exceeded|error)\b", re.IGNORECASE),
  re.compile(r"\bcontext\s+window\b", re.IGNORECASE),
  re.compile(r"\bmaximum\s+context\s+length\b", re.IGNORECASE),
  re.compile(r"\bprompt\s+too\s+long\b", re.IGNORECASE),
  re.compile(r"\btoo\s+many\s+(?:input\s+)?tokens\b", re.IGNORECASE),
  re.compile(r"\binput\s+(?:is\s+)?too\s+long\b", re.IGNORECASE),
  re.compile(r"\btoken\s+limit\b", re.IGNORECASE),
)

ThinkingMode = Literal["adaptive", "budget", "none"]


@dataclass
class ModelInfo:
  """Static or semi-static metadata about a model identifier."""

  id: str
  provider: str
  context_window: int = 200_000
  max_output_tokens: int = 16_384
  supports_thinking: bool = False
  supports_vision: bool = True
  supports_tool_use: bool = True
  supports_native_compaction: bool = False
  input_cost_per_mtok: float = 0.0
  output_cost_per_mtok: float = 0.0
  cache_read_cost_per_mtok: float = 0.0
  cache_write_cost_per_mtok: float = 0.0
  thinking_mode: ThinkingMode | None = None
  compat: dict[str, Any] | None = None

  def __post_init__(self) -> None:
    if self.thinking_mode is None:
      self.thinking_mode = "adaptive" if self.supports_thinking else "none"
    if self.thinking_mode != "none":
      self.supports_thinking = True


@dataclass
class StreamEvent:
  """Normalized provider stream event consumed by runners."""

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
  reasoning_tokens: int = 0
  provider_units: int = 0
  provider_unit_deltas: dict[str, int] | None = None
  cache_read_tokens: int = 0
  cache_creation_tokens: int = 0
  raw_block: Any = None
  caller: dict[str, Any] | None = None
  provider_reported_model: str | None = None


@dataclass
class CostEstimate:
  """Estimated request cost broken down by token category."""

  input_cost: float = 0.0
  output_cost: float = 0.0
  cache_read_cost: float = 0.0
  cache_write_cost: float = 0.0
  total: float = 0.0


def _status_code_from_exception(exc: Exception) -> int | None:
  status_code = getattr(exc, "status_code", None)
  if status_code is None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
  if isinstance(status_code, int):
    return status_code
  if isinstance(status_code, str) and status_code.isdigit():
    return int(status_code)
  return None


def _response_text(exc: Exception) -> str:
  response = getattr(exc, "response", None)
  text = getattr(response, "text", "") if response is not None else ""
  if isinstance(text, str):
    return text
  return ""


def _error_code_from_exception(exc: Exception) -> str | None:
  for attr in ("code", "error_code", "type"):
    value = getattr(exc, attr, None)
    if value:
      return str(value)
  body = _response_text(exc)
  for pattern in (
    re.compile(r'"code"\s*:\s*"([^"]+)"'),
    re.compile(r'"type"\s*:\s*"([^"]+)"'),
    re.compile(r"'code'\s*:\s*'([^']+)'"),
    re.compile(r"'type'\s*:\s*'([^']+)'"),
  ):
    match = pattern.search(body)
    if match:
      return match.group(1)
  return None


def _classify_provider_credential_failure(
  *,
  provider: str,
  exc: Exception,
) -> ProviderCredentialFailure | None:
  status_code = _status_code_from_exception(exc)
  message = " ".join(part for part in (str(exc), _response_text(exc)) if part).strip()
  error_code = _error_code_from_exception(exc)
  searchable = " ".join(part for part in (message, error_code or "") if part)

  if status_code is None:
    match = _STATUS_CODE_RE.search(searchable)
    if match:
      status_code = int(match.group(1))

  if any(pattern.search(searchable) for pattern in _BILLING_PATTERNS):
    return ProviderCredentialFailure(
      provider=provider,
      kind="billing",
      status_code=status_code,
      error_code=error_code,
      message=message,
    )

  if status_code in {401} or any(pattern.search(searchable) for pattern in _AUTH_PATTERNS):
    return ProviderCredentialFailure(
      provider=provider,
      kind="auth",
      status_code=status_code,
      error_code=error_code,
      message=message,
    )

  if status_code == 403:
    return ProviderCredentialFailure(
      provider=provider,
      kind="billing",
      status_code=status_code,
      error_code=error_code,
      message=message,
    )

  if status_code == 429 or any(pattern.search(searchable) for pattern in _RATE_LIMIT_PATTERNS):
    return ProviderCredentialFailure(
      provider=provider,
      kind="rate_limit",
      status_code=status_code,
      error_code=error_code,
      message=message,
    )

  return None


def _is_context_length_exception(exc: Exception) -> bool:
  error_code = _error_code_from_exception(exc)
  message = " ".join(part for part in (str(exc), _response_text(exc), error_code or "") if part)
  return any(pattern.search(message) for pattern in _CONTEXT_LENGTH_PATTERNS)


def truncate_to_last_compaction(
  messages: list[dict[str, Any]],
  *,
  compaction_as_text: bool = False,
) -> list[dict[str, Any]]:
  """Drop history already covered by the most recent server-side compaction block.

  Compaction blocks are produced by Anthropic's `compact_20260112` context
  management: the block's `content` is a model-generated summary that replaces
  everything before it. The compaction trigger evaluates raw submitted input
  tokens, not the post-replacement rendered context — resending the full
  pre-compaction history keeps every request above the trigger and the API
  re-summarizes from scratch on every turn. Submitting from the last compaction
  block onward (the documented "manually drop" client strategy) is what makes
  compaction actually reduce the next request's size.

  Providers whose APIs don't understand compaction blocks (anything
  non-Anthropic replaying an Anthropic-originated transcript) pass
  `compaction_as_text=True` to convert the anchor block into a plain text
  summary block instead of forwarding an unknown block type.
  """
  last_message_idx: int | None = None
  last_block_idx = 0
  for message_idx, message in enumerate(messages):
    if message.get("role") != "assistant":
      continue
    content = message.get("content")
    if not isinstance(content, list):
      continue
    for block_idx, block in enumerate(content):
      if isinstance(block, dict) and block.get("type") == "compaction":
        last_message_idx = message_idx
        last_block_idx = block_idx

  if last_message_idx is None:
    return messages

  anchor = dict(messages[last_message_idx])
  full_anchor_content = list(anchor.get("content") or [])
  anchor_content = full_anchor_content[last_block_idx:]
  if compaction_as_text:
    summary = str(anchor_content[0].get("content") or "")
    # Trailing separator: some providers concatenate adjacent text blocks with
    # no delimiter when building their request payload.
    anchor_content[0] = {
      "type": "text",
      "text": f"[Summary of the earlier conversation]\n{summary}\n\n",
    }
  anchor["content"] = anchor_content
  truncated = [anchor, *messages[last_message_idx + 1 :]]

  # If the anchor's dropped prefix contained tool_use blocks, their results in
  # the following user message would now be orphaned — drop those tool_results.
  dropped_tool_ids = {
    str(block.get("id", ""))
    for block in full_anchor_content[:last_block_idx]
    if isinstance(block, dict) and block.get("type") == "tool_use"
  }
  if dropped_tool_ids and len(truncated) > 1:
    follower = truncated[1]
    follower_content = follower.get("content")
    if follower.get("role") == "user" and isinstance(follower_content, list):
      kept_blocks = [
        block
        for block in follower_content
        if not (
          isinstance(block, dict)
          and block.get("type") == "tool_result"
          and str(block.get("tool_use_id", "")) in dropped_tool_ids
        )
      ]
      if kept_blocks:
        next_follower = dict(follower)
        next_follower["content"] = kept_blocks
        truncated[1] = next_follower
      else:
        truncated = [truncated[0], *truncated[2:]]

  return truncated


class ModelProvider:
  """Interface implemented by model-provider adapters.

  A provider is responsible for:

  - creating and closing API clients
  - validating model identifiers
  - translating gateway messages into provider request params
  - streaming normalized `StreamEvent` objects
  - estimating request cost
  """

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

  def resolve_effort(
    self,
    *,
    requested: ThinkingLevel,
    model: str,
    model_info: ModelInfo,
    max_tokens: int,
    **request_context: Any,
  ) -> EffortResolution:
    del model, max_tokens, request_context
    effective = requested if model_info.supports_thinking else ThinkingLevel.NONE
    return EffortResolution(
      requested=requested,
      effective=effective,
      thinking_enabled_effective=effective != ThinkingLevel.NONE,
      payload_fragments={},
    )

  def normalize_messages(self, messages: list[dict[str, Any]], model_info: ModelInfo) -> list[dict[str, Any]]:
    return list(messages)

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncIterator[StreamEvent]:
    raise NotImplementedError

  def is_retryable_error(self, exc: Exception) -> bool:
    return False

  def is_context_length_error(self, exc: Exception) -> bool:
    return False

  def classify_credential_failure(self, exc: Exception) -> ProviderCredentialFailure | None:
    return _classify_provider_credential_failure(provider=self.name, exc=exc)

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
