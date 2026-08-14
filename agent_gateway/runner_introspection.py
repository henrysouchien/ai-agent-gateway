from __future__ import annotations

import functools
import inspect
from typing import Any


_TRACEBACK_LOGGED_ATTR = "_gateway_traceback_logged"


def mark_exception_traceback_logged(exc: BaseException) -> None:
  """Mark exc as already logged with a traceback.

  Call this at any site that logs exc_info and then re-raises, so the terminal
  consumer (which converts the exception into a client-facing error event) logs
  the message without duplicating the traceback.

  Best-effort: an exception type that cannot take attributes stays unmarked and
  the consumer logs a duplicate traceback (never zero). Never raises — the guard
  is BaseException because the argument is a BaseException whose attribute hooks
  may themselves raise non-Exception errors, and this helper runs inside error
  handling where replacing the original failure is worse than swallowing.
  """
  try:
    setattr(exc, _TRACEBACK_LOGGED_ATTR, True)
  except BaseException:
    pass


def exception_traceback_already_logged(exc: BaseException) -> bool:
  """True iff mark_exception_traceback_logged was (effectively) applied to exc.

  Never raises (BaseException-guarded, same rationale as the marker) —
  pathological attribute access reads as "not marked", which fails toward a
  duplicate traceback, not a lost one.
  """
  try:
    return bool(getattr(exc, _TRACEBACK_LOGGED_ATTR, False))
  except BaseException:
    return False


def derive_sub_agent_id(parent_session: Any, call_index: int) -> str:
  parent_sid = str(getattr(parent_session, "session_id", parent_session) or "")
  return f"sub{int(call_index)}:{parent_sid}"


def format_exc(exc: BaseException) -> str:
  parts = [f"{type(exc).__name__}: {repr(exc)}"]
  seen = {id(exc)}
  cause = exc.__cause__
  while cause is not None and id(cause) not in seen:
    parts.append(f"caused by {type(cause).__name__}: {repr(cause)}")
    seen.add(id(cause))
    cause = cause.__cause__
  return " | ".join(parts)


def detect_keyword_param(fn, param_name: str) -> bool:
  """True iff fn accepts `param_name` as a keyword argument.

  Returns True for POSITIONAL_OR_KEYWORD, KEYWORD_ONLY, and **kwargs.
  Returns False for None, positional-only params, and uninspectable callables.
  """
  if fn is None:
    return False
  if isinstance(fn, functools.partial):
    return False
  try:
    sig = inspect.signature(fn)
  except (TypeError, ValueError):
    return False
  for name, param in sig.parameters.items():
    if param.kind is inspect.Parameter.VAR_KEYWORD:
      return True
    if name == param_name and param.kind in (
      inspect.Parameter.POSITIONAL_OR_KEYWORD,
      inspect.Parameter.KEYWORD_ONLY,
    ):
      return True
  return False


def detect_user_id_param(fn) -> bool:
  """True iff fn accepts user_id as a keyword argument."""
  return detect_keyword_param(fn, "user_id")
