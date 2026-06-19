import functools
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import _derive_sub_agent_id as package_derive_sub_agent_id  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.runner_introspection import (  # noqa: E402
  derive_sub_agent_id,
  detect_user_id_param,
  format_exc,
)
from agent_gateway.sdk_runner import _detect_user_id_param as sdk_detect_user_id_param  # noqa: E402
from agent_gateway.sub_agent import _derive_sub_agent_id as sub_agent_derive_sub_agent_id  # noqa: E402


def test_runner_preserves_introspection_helper_aliases() -> None:
  assert gateway_runner._derive_sub_agent_id is derive_sub_agent_id
  assert package_derive_sub_agent_id is derive_sub_agent_id
  assert sub_agent_derive_sub_agent_id is derive_sub_agent_id
  assert gateway_runner._format_exc is format_exc
  assert gateway_runner._detect_user_id_param is detect_user_id_param
  assert sdk_detect_user_id_param is detect_user_id_param


def test_derive_sub_agent_id_accepts_session_objects_and_plain_ids() -> None:
  assert derive_sub_agent_id(SimpleNamespace(session_id="parent-session"), 3) == "sub3:parent-session"
  assert derive_sub_agent_id("raw-parent", "4") == "sub4:raw-parent"
  assert derive_sub_agent_id(SimpleNamespace(session_id=""), 0) == "sub0:"


def test_format_exc_includes_cause_chain_once() -> None:
  try:
    try:
      raise ValueError("root cause")
    except ValueError as exc:
      raise RuntimeError("wrapper") from exc
  except RuntimeError as exc:
    formatted = format_exc(exc)

  assert "RuntimeError: RuntimeError('wrapper')" in formatted
  assert "caused by ValueError: ValueError('root cause')" in formatted


def test_detect_user_id_param_accepts_keyword_shapes() -> None:
  def positional_or_keyword(user_id):
    return user_id

  def keyword_only(*, user_id):
    return user_id

  def arbitrary_keywords(**kwargs):
    return kwargs

  assert detect_user_id_param(positional_or_keyword) is True
  assert detect_user_id_param(keyword_only) is True
  assert detect_user_id_param(arbitrary_keywords) is True


def test_detect_user_id_param_rejects_unsupported_shapes() -> None:
  def positional_only(user_id, /):
    return user_id

  def no_user_id(value):
    return value

  assert detect_user_id_param(None) is False
  assert detect_user_id_param(positional_only) is False
  assert detect_user_id_param(no_user_id) is False
  assert detect_user_id_param(functools.partial(no_user_id, "fixed")) is False
  assert detect_user_id_param(42) is False
