from __future__ import annotations


MAX_NOTIFICATIONS_PER_TURN = 5

# Top-level interactive runs retain a bounded continuation guard. Child/workflow
# logical responses are unbounded here: max_tokens segments are provider-call
# mechanics, while existing wall-clock timeout, cancellation, and operator
# controls provide operational runaway protection.
MAX_TOKENS_CONTINUATIONS = 3
MAX_TOKENS_NUDGE = (
  "[System: Your previous response hit the output-token limit and was truncated; "
  "any partial tool call was discarded. Continue the task now with a tool-first "
  "response: if a required tool/report/persist door is available, call it now with "
  "the smallest valid JSON payload. Trim verbose rationale fields, omit optional "
  "narrative, and split only when the tool contract requires it. Do not spend "
  "another turn on hidden analysis or restate prior reasoning.]"
)


__all__ = [
  "MAX_NOTIFICATIONS_PER_TURN",
  "MAX_TOKENS_CONTINUATIONS",
  "MAX_TOKENS_NUDGE",
]
