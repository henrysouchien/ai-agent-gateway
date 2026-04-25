# Package Update Checklist

Reference this checklist when adding features, fixing bugs, or changing the public API of `ai-agent-gateway`.

## Always (every change)

- [ ] **Tests pass** — `pytest packages/agent-gateway/tests/` all green
- [ ] **Existing consumer tests pass** — `pytest tests/test_code_execute.py tests/test_tool_dispatcher.py tests/test_channel_registry.py tests/test_run_agent.py` etc.
- [ ] **New code has docstrings** — every new public function/class gets a docstring at write time

## When adding new public API symbols

- [ ] **`__init__.py` exports** — add to imports and `__all__`
- [ ] **`docs/api-reference.md`** — add entry in the appropriate category
- [ ] **Docstring on the symbol** — params, return type, one-line example if applicable

## When adding new features

- [ ] **`docs/architecture.md`** — update if the feature introduces a new concept, flow, or mental model change
- [ ] **Tests for the feature** — unit tests in `packages/agent-gateway/tests/`, integration tests if it touches consumer wiring
- [ ] **Example update or new example** — if the feature is user-facing and changes how someone would use `create_agent()` or `create_gateway_app()`

## When adding new SSE events or endpoints

- [ ] **`docs/http-api.md`** — add the event type with payload schema, or the endpoint with request/response

## When changing `create_agent()` signature

- [ ] **`docs/quickstart.md`** — update if the quickstart flow is affected
- [ ] **README progressive examples** — update the relevant tier if the API changed
- [ ] **`tests/test_easy.py`** — add test coverage for new params

## When changing `create_gateway_app()` or core classes

- [ ] **`docs/architecture.md`** — update the relevant section
- [ ] **`docs/api-reference.md`** — update field docs for changed dataclasses

## Deprecation log

- `send_prompt(on_usage=...)` legacy 4-int callbacks are deprecated in favor of `UsageEvent` callbacks (introduced 0.11.1). Target removal: `v0.12.0`.
- Reconciler deployments require SQLite `3.35.0+` for `UPDATE ... RETURNING`.

## What NOT to update for every feature

- **README** — only update for major capability changes that alter the positioning or add a new tier. Incremental enhancements to existing features (e.g., background mode for sub-agents) don't need README changes.
- **`docs/comparison.md`** — only update if a feature changes our competitive positioning
- **`CONTRIBUTING.md`** — only update if dev workflow changes

## Publish

- [ ] **Commit** the feature + doc updates together
- [ ] **Bump version** — patch for fixes, minor for features, major for breaking changes
- [ ] **`scripts/publish_agent_gateway.sh --patch|--minor|--major --yes`**
- [ ] **`pip install --upgrade ai-agent-gateway`** locally after publish
