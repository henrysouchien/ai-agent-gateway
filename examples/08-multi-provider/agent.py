from agent_gateway import create_agent


app = create_agent(
  "You are a concise assistant running through the first-party OpenAI Responses API.",
  provider="openai",
  model="gpt-5.6",
  max_turns=6,
  valid_api_keys={"demo-key"},
  cors_origins=["*"],
)
