from pathlib import Path

from agent_gateway import create_agent


SKILLS_DIR = Path(__file__).parent / "skills"

app = create_agent(
  "When a task has a clear research phase or summarization phase, use run_agent with the named skills "
  "instead of doing everything in one long turn.",
  skills_dir=SKILLS_DIR,
)
