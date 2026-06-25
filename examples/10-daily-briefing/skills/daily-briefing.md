---
name: daily-briefing
agent_callable: true
agent_description: Turn provided source notes into a concise daily briefing with actions and risks.
persist_state: true
max_turns: 4
---

Create a compact daily briefing from the material provided by the parent agent.

Use this structure:

1. `## HEADLINES` with 2-4 bullets.
2. `## WATCH ITEMS` for unresolved risks, blockers, or follow-ups.
3. `## NEXT ACTIONS` with concrete owners if the source text names them; otherwise leave owners unassigned.

Do not invent new facts. If an item is an inference from multiple source notes, label it as an inference.

When the briefing is complete, append:

## STATE_UPDATE_JSON
```json
{"last_briefing_topic": "daily briefing", "last_output_shape": "headlines_watch_items_next_actions"}
```
