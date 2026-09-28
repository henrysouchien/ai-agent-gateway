from __future__ import annotations


MAX_NOTIFICATIONS_PER_TURN = 5

# Top-level interactive runs retain a bounded continuation guard. Child/workflow
# logical responses are unbounded here: max_tokens segments are provider-call
# mechanics, while existing wall-clock timeout, cancellation, and operator
# controls provide operational runaway protection.
MAX_TOKENS_CONTINUATIONS = 3
MAX_TOKENS_NUDGE = (
  "[System: Your previous response was cut off at the output-token limit, and "
  "any partial tool call in it was discarded. Continue the task: if you were "
  "making a tool call, re-issue that call complete, with every section and field "
  "you decided on, as the first thing in your response. Do not restate prior "
  "reasoning.]"
)


__all__ = [
  "MAX_NOTIFICATIONS_PER_TURN",
  "MAX_TOKENS_CONTINUATIONS",
  "MAX_TOKENS_NUDGE",
]
