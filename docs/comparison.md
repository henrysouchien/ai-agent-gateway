# Comparison

This page compares `ai-agent-gateway` to a few adjacent frameworks with different priorities.

These projects evolve quickly. Treat this page as a directional architecture comparison, not a substitute for checking the current upstream docs before a long-term platform decision.

## At A Glance

| Category | ai-agent-gateway | LangGraph | LangChain | CrewAI | mcp-agent |
| --- | --- | --- | --- | --- | --- |
| Primary purpose | Deploying agents as services | Building stateful workflow graphs | Assembling LLM applications from reusable components | Coordinating role-based multi-agent teams | Orchestrating MCP-centric agent workflows |
| Agent logic | Model-driven prompts plus tools | Code-defined graph nodes and edges | Code-defined chains, agents, and integrations | Code-defined roles, tasks, and crews | Code-defined workflows and MCP actions |
| Tool system | MCP-native plus local Python handlers | Bring your own adapters and tool nodes | Bring your own adapters and integration wrappers | Custom tool abstractions | MCP-native |
| Server/runtime in core | FastAPI + SSE server in core | Bring your own server or use LangGraph deployment products | LangServe is separate from core LangChain | Bring your own server/runtime | Bring your own server/runtime |
| Sessions/auth in core | JWT sessions in core | Bring your own | Bring your own | Bring your own | Bring your own |
| Human-in-the-loop approval | Built into tool dispatch | Interrupt/checkpoint patterns available | Not a core runtime primitive | Human input patterns are available, but not as a gateway-native approval loop | Not a core runtime primitive |
| Code execution in core | Docker sandbox plus subprocess fallback | Not in core | Available through integrations/community patterns | Not in core | Not in core |
| Multi-provider story | Anthropic via `create_agent()`, broader provider control via `create_gateway_app()` | Bring your own model layer | Broad model integration ecosystem | Bring your own model layer | Bring your own model layer |
| Best fit | Shipping a chat-facing agent backend quickly | Explicit workflow control flow and durable graph logic | LLM application composition and integration breadth | Team-style simulations and role/task collaboration | MCP-heavy automation and orchestration |

## How To Read The Differences

### ai-agent-gateway

Best when you want to deploy an agent that users or clients talk to over HTTP and you do not want to build:

- session management
- SSE streaming
- approval endpoints
- tool dispatch plumbing
- basic auth and runtime envelopes

It is especially strong when your agent logic is still mostly model-driven and your main challenge is packaging that logic into a service.

### LangGraph

Best when the hard part of your problem is explicit workflow control flow:

- branching
- retries
- durable checkpoints
- human interrupts inside a graph
- multi-step state transitions you want to own in code

LangGraph is a better fit when you want the workflow itself to be the product.

### LangChain

Best when you primarily want a library of model, retriever, prompt, and tool-building blocks.

LangChain is broad and integration-heavy. It is a good fit when you want composition primitives and ecosystem coverage more than an opinionated agent server runtime.

### CrewAI

Best when you want a role/task/crew mental model for multi-agent collaboration.

It is a good fit for simulations, delegated task structures, or team-style workflows where explicit roles matter more than exposing one durable chat server.

### mcp-agent

Best when MCP is the center of gravity and you want orchestration around MCP tools and servers.

It is a good fit when the tool layer is the main product surface and the HTTP/session layer is secondary.

## When To Use Us

Use `ai-agent-gateway` when:

- you want an agent behind a stable HTTP API
- your frontend needs SSE streaming out of the box
- you want session-scoped approvals and session-scoped tool state
- you want to start with a system prompt and grow into MCP, local tools, skills, and code execution without changing runtimes immediately
- you want the upgrade path from `create_agent()` to `create_gateway_app()`

## When Not To Use Us

Do not reach for `ai-agent-gateway` first when:

- you need explicit graph orchestration as the primary abstraction
- you are building a one-off script, notebook, or batch job that does not need a reusable server
- your main need is a broad integration library rather than a deployment runtime
- you want team/crew abstractions to be the core programming model

## Can You Use Both?

Yes.

These tools are often complementary rather than mutually exclusive. A common split is:

- LangGraph for complex workflow logic
- LangChain for integration helpers
- `ai-agent-gateway` for the HTTP/session/approval/SSE serving layer

If your users need a stable agent API, the gateway can be the outer runtime even when the inner decision logic comes from somewhere else.
