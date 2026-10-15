from __future__ import annotations

from agent_gateway.providers.base import ModelInfo
from agent_gateway.providers.ollama import OllamaLocalProvider
from agent_gateway.runner_streaming import (
  LOCAL_PREFILL_TOKENS_PER_SECOND,
  STREAM_STALL_TIMEOUT,
  STREAM_THINKING_STALL_TIMEOUT,
  effective_stream_stall_timeout,
)
from model_authority.thinking import EffortResolution, ThinkingLevel

_NO_THINKING = EffortResolution(ThinkingLevel.NONE, ThinkingLevel.NONE, False, {})
_THINKING = EffortResolution(ThinkingLevel.HIGH, ThinkingLevel.HIGH, True, {})


def _bound(model_info: ModelInfo, effort: EffortResolution) -> float:
  return effective_stream_stall_timeout(
    None,
    config={},
    model_info=model_info,
    max_tokens=4096,
    effort_resolution=effort,
  )


def test_a_cloud_transport_keeps_the_hung_peer_bound() -> None:
  info = ModelInfo(id="cloud", provider="anthropic", context_window=1_000_000)
  assert _bound(info, _NO_THINKING) == STREAM_STALL_TIMEOUT
  assert _bound(info, _THINKING) == STREAM_THINKING_STALL_TIMEOUT


def test_a_local_prefill_transport_scales_the_bound_from_its_context_window() -> None:
  # qwen3.8:27b on this Mac: 262144-token window. 25517 prompt tokens took 65 s to a first
  # chunk and 60945 took up to 232 s, past the 60 s bound this scaling replaces.
  info = ModelInfo(id="qwen", provider="ollama", context_window=262_144, first_progress_after_prefill=True)
  bound = _bound(info, _NO_THINKING)
  assert bound == 262_144 / LOCAL_PREFILL_TOKENS_PER_SECOND
  assert bound > 65
  # Effort does not lengthen it further; it is never shorter than the request's own bound.
  assert _bound(info, _THINKING) == max(STREAM_THINKING_STALL_TIMEOUT, bound)


def test_a_small_local_window_never_drops_below_the_base_bound() -> None:
  info = ModelInfo(id="tiny", provider="ollama", context_window=2_048, first_progress_after_prefill=True)
  assert _bound(info, _NO_THINKING) == STREAM_STALL_TIMEOUT


def test_an_explicit_per_run_bound_still_wins() -> None:
  info = ModelInfo(id="qwen", provider="ollama", context_window=262_144, first_progress_after_prefill=True)
  assert effective_stream_stall_timeout(5.0, config={}, model_info=info, max_tokens=4096) == 5.0


def test_packaged_local_prefill_follows_the_entry_route() -> None:
  # admitted_entry resolves the model id to the packaged row. A local tag
  # prefills on this host; a daemon-proxied cloud tag prefills on ollama.com.
  provider = OllamaLocalProvider()
  local = provider.get_model_info("qwen3.8:27b")
  cloud = provider.get_model_info("kimi-k3:cloud")
  assert local.first_progress_after_prefill is True
  assert cloud.first_progress_after_prefill is False
  assert _bound(cloud, _NO_THINKING) == STREAM_STALL_TIMEOUT == 60
