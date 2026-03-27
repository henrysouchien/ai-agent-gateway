import os

from agent_gateway import create_agent


base_url = os.getenv("OPENAI_BASE_URL", "").strip()
provider_config = {"base_url": base_url} if base_url else None

app = create_agent(
  "You are a concise assistant running through the OpenAI provider adapter.",
  provider="openai",
  model="gpt-4o",
  provider_config=provider_config,
  max_turns=6,
  valid_api_keys={"demo-key"},
  cors_origins=["*"],
)
