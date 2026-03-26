from claude_gateway import create_agent

app = create_agent(
  "You can use filesystem tools to inspect and edit files in this example directory. "
  "Before changing a file, briefly explain what you are about to do.",
  mcp_servers={
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
    }
  },
)
