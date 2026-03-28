# Autonomous Run Example

This example shows the new `run_autonomous_sync()` entry point for a headless run with local tools, state persistence, and optional Telegram delivery.

## Run

Set credentials for your provider first. For Anthropic:

```bash
export ANTHROPIC_API_KEY=...
```

Telegram delivery is optional:

```bash
export TELEGRAM_BOT_TOKEN=...
export TELEGRAM_CHAT_ID=...
```

Then run:

```bash
python agent.py
```

The script writes state to `examples/09-autonomous/state/state.json` and prints the final response to stdout.

## Heartbeat Mode

For persistent agents that check in periodically, wrap `run_autonomous()` with `HeartbeatLoop`:

```python
from functools import partial
from agent_gateway import run_autonomous, HeartbeatLoop, HeartbeatConfig

loop = HeartbeatLoop(
    run_fn=partial(run_autonomous,
        system_prompt="Check HEARTBEAT.md. If nothing needs attention, reply HEARTBEAT_OK.",
        initial_message="Check if anything needs attention.",
        model="claude-sonnet-4-6",
        tool_handlers={...},
    ),
    config=HeartbeatConfig(
        interval_seconds=1800,       # 30 minutes
        active_hours=(6, 22),        # 6am-10pm
        timezone="America/New_York",
    ),
    on_alert=my_delivery_callback,   # called only when agent reports something
)
await loop.start()  # blocks until loop.stop()
```

The loop automatically suppresses `HEARTBEAT_OK` responses, skips ticks outside active hours, and applies exponential backoff on errors.
