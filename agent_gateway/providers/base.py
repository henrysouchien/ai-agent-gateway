from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import re
import time
from typing import Any, AsyncIterator, Literal, Mapping

from ..auth import ProviderCredentialFailure
from model_authority.current import INITIAL_MODEL_REGISTRY, RATE_TABLES
from model_authority.registry import AdapterRouteSupport, ModelRegistryEntry
from model_authority.rates import ContextRateTier, RateTable, UnknownModelError, provider_rate_table
from model_authority.schema import FamilyCompat
from model_authority.thinking import EffortResolution, ThinkingLevel

log = logging.getLogger(__name__)


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
  # A 403 whose organization refuses OAuth is a property of this credential's
  # account, not of the request: a sibling in another organization answers it.
  re.compile(r"\boauth_not_allowed_for_organization\b", re.IGNORECASE),
)
_RATE_LIMIT_PATTERNS = (
  re.compile(r"\brate(?:\s+limit(?:ed|ing)?|\s+limited)\b", re.IGNORECASE),
  re.compile(r"\btoo many requests\b", re.IGNORECASE),
)
# Anthropic's unified subscription limiter reports one status and one reset per
# window; the status names whether that window is still serving requests.
_LIMIT_WINDOW_PROJECTIONS = (
  ("rate_limit_5h_status", "rate_limit_5h_reset"),
  ("rate_limit_7d_status", "rate_limit_7d_reset"),
)
# 2001-09-09T01:46:40Z: above it a projected number is an instant, below it a delay.
_EPOCH_SECONDS_FLOOR = 1_000_000_000.0
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
  """Request facts for one model id, built from its model-authority entry.

  Limits and ``compat`` come from the entry's family compat block, efforts
  from its ``supported_efforts``, prices from the authority's rate table.
  """

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
  compat: FamilyCompat | None = None
  # The entry's supported efforts in canonical ``ThinkingLevel`` order.
  effort_values: tuple[str, ...] = ()
  rate_tiers: tuple[ContextRateTier, ...] = ()

  def __post_init__(self) -> None:
    if self.thinking_mode is None:
      self.thinking_mode = "adaptive" if self.supports_thinking else "none"
    if self.thinking_mode != "none":
      self.supports_thinking = True


def registry_entry_for_model(support: AdapterRouteSupport, model_id: str) -> ModelRegistryEntry | None:
  """The model-authority entry one adapter executes for a provider model id.

  Only entries the adapter's own route-support declaration covers qualify, so
  an upstream id the authority also lists for another process's adapter never
  shapes this adapter's requests.  An exact ``upstream_model`` match wins over
  a ``reported_identities`` alias; iteration is key-ordered for determinism.
  """
  alias_match: ModelRegistryEntry | None = None
  for key in sorted(INITIAL_MODEL_REGISTRY.models):
    entry = INITIAL_MODEL_REGISTRY.models[key]
    if not support.supports(entry):
      continue
    if entry.upstream_model == model_id:
      return entry
    if alias_match is None and model_id in entry.reported_identities:
      alias_match = entry
  return alias_match


def admitted_entry(support: AdapterRouteSupport, model: str) -> tuple[str, ModelRegistryEntry]:
  """The stripped model id and its entry; a model id with no entry is not callable."""
  model_id = str(model or "").strip()
  if not model_id:
    raise ValueError("Model is required")
  entry = registry_entry_for_model(support, model_id)
  if entry is None:
    raise ValueError(
      f"the model authority admits no {support.adapter} entry for model {model_id!r}"
    )
  return model_id, entry


def registry_effort_values(entry: ModelRegistryEntry) -> tuple[str, ...]:
  """The entry's supported efforts in canonical ``ThinkingLevel`` order."""
  return tuple(
    level.value for level in ThinkingLevel if level.value in entry.supported_efforts
  )


def authority_rate_table(provider: str, override: RateTable | None = None) -> RateTable:
  """This provider's authority rate table, with an explicit override's rows first."""
  return provider_rate_table(provider, RATE_TABLES, override)


def model_info_from_entry(
  model_id: str,
  entry: ModelRegistryEntry,
  rate_table: RateTable,
  *,
  supports_thinking: bool,
  thinking_mode: ThinkingMode | None = None,
  supports_native_compaction: bool = False,
) -> ModelInfo:
  """Build ``ModelInfo`` from the entry, its compat limits and the rate table's prices."""
  compat = entry.compat
  info = ModelInfo(
    id=model_id,
    provider=entry.provider,
    context_window=compat.context_window,
    max_output_tokens=compat.max_output_tokens,
    supports_thinking=supports_thinking,
    supports_vision="vision" in entry.features,
    supports_tool_use="tools" in entry.features,
    supports_native_compaction=supports_native_compaction,
    thinking_mode=thinking_mode,
    compat=compat,
    effort_values=registry_effort_values(entry),
  )
  try:
    rates = rate_table.lookup(entry.provider, model_id)
  except UnknownModelError:
    log.warning("%s model %r has no rate row; using zero-cost estimates", entry.provider, model_id)
    return info
  info.input_cost_per_mtok = rates.input_cost_per_mtok
  info.output_cost_per_mtok = rates.output_cost_per_mtok
  info.cache_read_cost_per_mtok = rates.cache_read_cost_per_mtok
  info.cache_write_cost_per_mtok = rates.cache_write_cost_per_mtok
  info.rate_tiers = rates.tiers
  return info


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
  stop_details: dict[str, Any] | None = None
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


def _projected_epoch_seconds(value: Any, *, now: float) -> float | None:
  """Read one projected reset field as epoch seconds.

  Providers project two shapes: a delay in seconds (`retry-after`) and an
  absolute reset instant (`anthropic-ratelimit-unified-*-reset`, RFC 3339 or
  epoch seconds). Both are the limiter's own words; this only converts them.
  """

  if isinstance(value, bool) or value is None:
    return None
  text = str(value).strip()
  if not text:
    return None
  try:
    numeric = float(text)
  except ValueError:
    try:
      parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
      return None
    if parsed.tzinfo is None:
      parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()
  # A value small enough to predate this decade is a delay, not an instant.
  return numeric if numeric > _EPOCH_SECONDS_FLOOR else now + numeric


def _credential_reset_at(exc: Exception, *, now: float | None = None) -> float | None:
  """When the limiter said this credential becomes usable again.

  Reads the rate-limit projection the provider layer already attached to the
  exception (`api/credentials.py` `_anthropic_rate_limit_projection`); it never
  re-reads response headers. A limit window is only counted while its own
  status says it is not serving requests, so a burst 429 does not park a
  credential until an unrelated weekly window rolls over.
  """

  moment = time.time() if now is None else now
  candidates: list[float] = []
  retry_after = _projected_epoch_seconds(getattr(exc, "retry_after", None), now=moment)
  if retry_after is not None:
    candidates.append(retry_after)
  for status_field, reset_field in _LIMIT_WINDOW_PROJECTIONS:
    status = str(getattr(exc, status_field, "") or "").strip().lower()
    if not status or status.startswith("allow"):
      continue
    window_reset = _projected_epoch_seconds(getattr(exc, reset_field, None), now=moment)
    if window_reset is not None:
      candidates.append(window_reset)
  return max(candidates) if candidates else None


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
      reset_at=_credential_reset_at(exc),
    )

  if status_code in {401} or any(pattern.search(searchable) for pattern in _AUTH_PATTERNS):
    return ProviderCredentialFailure(
      provider=provider,
      kind="auth",
      status_code=status_code,
      error_code=error_code,
      message=message,
      reset_at=_credential_reset_at(exc),
    )

  if status_code == 403:
    return ProviderCredentialFailure(
      provider=provider,
      kind="billing",
      status_code=status_code,
      error_code=error_code,
      message=message,
      reset_at=_credential_reset_at(exc),
    )

  if status_code == 429 or any(pattern.search(searchable) for pattern in _RATE_LIMIT_PATTERNS):
    return ProviderCredentialFailure(
      provider=provider,
      kind="rate_limit",
      status_code=status_code,
      error_code=error_code,
      message=message,
      reset_at=_credential_reset_at(exc),
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

  @classmethod
  def adapter_route_support(cls) -> AdapterRouteSupport | None:
    """Declare the protocol facts this adapter implementation actually provides.

    Installed capability adapters return their adapter id, credential provider
    family, implemented protocol profiles, and supported deployment routes.
    The declaration is a statement about the installed code — plan §7: adapter
    declarations are protocol facts, not a second model catalog — and startup
    closure admits registry entries against it.  ``None`` means this class
    declares no capability adapter (base classes and app-owned stubs); a
    deployment-supplied ``capability_adapter_resolver`` then vouches for any
    adapter it maps onto such a class.
    """

    return None

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

  def stream(self, client: Any, params: dict[str, Any]) -> AsyncIterator[StreamEvent]:
    raise NotImplementedError

  def is_retryable_error(self, exc: Exception) -> bool:
    return False

  def is_context_length_error(self, exc: Exception) -> bool:
    return False

  def classify_credential_failure(self, exc: Exception) -> ProviderCredentialFailure | None:
    return _classify_provider_credential_failure(provider=self.name, exc=exc)

  def next_credential(
    self,
    config: Mapping[str, Any],
    failure: ProviderCredentialFailure,
  ) -> dict[str, Any] | None:
    """Credential material replacing the one this failure made unusable.

    A provider that keeps several interchangeable same-mode credentials
    returns the next usable one so the bound model keeps serving the run.
    ``None`` means this provider has nothing to swap in and the caller's
    existing retry stands.
    """

    del config, failure
    return None

  def estimate_cost(
    self,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
  ) -> CostEstimate:
    info = self.get_model_info(model)
    # Normalized input/cache counts are disjoint; tier thresholds use the full prompt.
    prompt_tokens = input_tokens + cache_read_tokens + cache_creation_tokens
    prices: ModelInfo | ContextRateTier = info
    for tier in info.rate_tiers:
      if prompt_tokens >= tier.min_input_tokens:
        prices = tier
        break
    input_cost = input_tokens * prices.input_cost_per_mtok / 1_000_000
    output_cost = output_tokens * prices.output_cost_per_mtok / 1_000_000
    cache_read_cost = cache_read_tokens * prices.cache_read_cost_per_mtok / 1_000_000
    cache_write_cost = cache_creation_tokens * prices.cache_write_cost_per_mtok / 1_000_000
    return CostEstimate(
      input_cost=input_cost,
      output_cost=output_cost,
      cache_read_cost=cache_read_cost,
      cache_write_cost=cache_write_cost,
      total=input_cost + output_cost + cache_read_cost + cache_write_cost,
    )
