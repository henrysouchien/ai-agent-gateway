from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, AsyncIterator

from ..auth import ProviderCredentialFailure


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

class ThinkingLevel(str, Enum):
  """Provider-agnostic reasoning intensity hint."""

  NONE = "none"
  MINIMAL = "minimal"
  LOW = "low"
  MEDIUM = "medium"
  HIGH = "high"
  MAX = "max"


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
  input_cost_per_mtok: float = 0.0
  output_cost_per_mtok: float = 0.0
  cache_read_cost_per_mtok: float = 0.0
  cache_write_cost_per_mtok: float = 0.0
  compat: dict[str, Any] | None = None


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
  cache_read_tokens: int = 0
  cache_creation_tokens: int = 0
  raw_block: Any = None
  caller: dict[str, Any] | None = None


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

  def normalize_messages(self, messages: list[dict[str, Any]], model_info: ModelInfo) -> list[dict[str, Any]]:
    return list(messages)

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncIterator[StreamEvent]:
    raise NotImplementedError

  def is_retryable_error(self, exc: Exception) -> bool:
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
