from claude_gateway import CodeExecutionConfig, create_agent

app = create_agent(
  "Use code_execute for calculations, quick scripts, and plots. "
  "Prefer code execution when math or data transformation would be more reliable than mental arithmetic.",
  code_execution=True,
  code_execution_config=CodeExecutionConfig(
    default_timeout_ms=20_000,
    max_timeout_ms=60_000,
  ),
)
