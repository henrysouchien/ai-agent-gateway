# Autonomous Run Example

This example shows the execution-only `run_autonomous_sync()` entry point for a
headless run with local tools, state persistence, and optional Telegram
delivery. `agent.py` constructs an explicit prebound fixture so the executor
receives one exact `session.driver` bind, provider adapter, and immutable
credential snapshot. Production callers should obtain those values from the
server-owned capability resolver rather than construct a bind inline.

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
from agent_gateway import (
    BoundCapabilityExecution,
    GatewaySession,
    HeartbeatConfig,
    HeartbeatLoop,
    run_autonomous,
)

# Resolve these once through the server-owned capability resolver. The bind's
# run_mode must be "autonomous" or "cron", and the transport must be native.
capability_execution: BoundCapabilityExecution
session: GatewaySession
capability_execution, session = prepare_autonomous_execution()

loop = HeartbeatLoop(
    run_fn=partial(run_autonomous,
        system_prompt="Check HEARTBEAT.md. If nothing needs attention, reply HEARTBEAT_OK.",
        initial_message="Check if anything needs attention.",
        capability_execution=capability_execution,
        session=session,
        tool_handlers={...},
        user_id="heartbeat-agent",
        billing_mode="byok",
        rate_table_version="current",
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
