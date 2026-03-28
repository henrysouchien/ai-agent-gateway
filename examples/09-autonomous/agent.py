import os
from pathlib import Path

from agent_gateway import DeliveryConfig, run_autonomous_sync


BASE_DIR = Path(__file__).parent
STATE_DIR = BASE_DIR / "state"


async def read_status(_tool_input, **_kwargs):
  return {
    "service": "agent-gateway",
    "status": "green",
    "checked_by": "09-autonomous example",
  }, None


TOOL_DEFINITIONS = [
  {
    "name": "read_status",
    "description": "Read the current service status before sending the final summary.",
    "input_schema": {
      "type": "object",
      "properties": {},
    },
  },
]


if __name__ == "__main__":
  result = run_autonomous_sync(
    "You are a concise operations assistant. Always call read_status before you reply.",
    "Check the current service status and send a short summary.",
    tool_handlers={"read_status": read_status},
    tool_definitions=TOOL_DEFINITIONS,
    state_dir=STATE_DIR,
    delivery=DeliveryConfig(
      telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or None,
      telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip() or None,
      telegram_label="Agent Gateway autonomous example",
    ),
  )
  print(result.response)
