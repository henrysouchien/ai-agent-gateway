from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
  from mcp.client.session import ClientSession
  from mcp.client.stdio import StdioServerParameters, stdio_client
  MCP_IMPORT_ERROR: Exception | None = None
except Exception as exc:
  ClientSession = Any  # type: ignore[assignment]
  StdioServerParameters = Any  # type: ignore[assignment]
  MCP_IMPORT_ERROR = exc

  def stdio_client(*args: Any, **kwargs: Any) -> Any:
    _ = args, kwargs
    raise RuntimeError(f"MCP client runtime unavailable: {MCP_IMPORT_ERROR}")


log = logging.getLogger("agent_gateway.mcp_client")
_UNSET = object()


def _classify_exception(exc: Exception, msg: str) -> str:
  lower = msg.lower()
  if isinstance(exc, asyncio.TimeoutError) or "timeout" in lower or "timed out" in lower:
    return "timeout"
  if "connection" in lower or "refused" in lower:
    return "connection_error"
  return "unknown"


def _classify_mcp_error(message: str) -> str:
  lower = message.lower()
  if "not found" in lower or "no filing" in lower or "no data" in lower:
    return "not_found"
  if "parse" in lower or "invalid" in lower or "malformed" in lower:
    return "parse_error"
  if "timeout" in lower or "timed out" in lower:
    return "timeout"
  return "unknown"


@dataclass
class _ServerState:
  name: str
  session: ClientSession
  exit_contexts: List[Any]
  tool_definitions: List[Dict[str, Any]]
  tool_names: Set[str]


class McpClientManager:
  """Manage stdio MCP server lifecycles and tool routing.

  The manager can load servers from inline config, from `~/.claude.json`, or
  from an alternate config path. On startup it connects to each allowed stdio
  server, lists its tools, filters name collisions, and exposes a merged tool
  catalog to the runner.

  `inline_servers` is the easiest way to ship self-contained examples because it
  avoids any dependency on the user's Claude desktop config.
  """

  def __init__(
    self,
    allowed_servers: Set[str] | None = None,
    builtin_tool_names: Set[str] | None = None,
    config_path: Path | str | None | object = _UNSET,
    inline_servers: Dict[str, Dict[str, Any]] | None = None,
    timeout_overrides: Dict[str, int] | None = None,
    startup_timeout: int = 15,
    default_tool_timeout: int = 30,
  ) -> None:
    self._lock = asyncio.Lock()
    self._started = False
    self._servers: Dict[str, _ServerState] = {}
    self._tool_definitions: List[Dict[str, Any]] = []
    self._tool_to_server: Dict[str, str] = {}
    self._mcp_tool_names: Set[str] = set()
    self._allowed_servers = allowed_servers
    self._builtin_tool_names = set(builtin_tool_names or set())
    self._inline_servers = dict(inline_servers or {})
    if config_path is _UNSET:
      self._config_path: Path | None = Path.home() / ".claude.json"
    elif config_path is None:
      self._config_path = None
    else:
      self._config_path = Path(config_path).expanduser()
    self._timeout_overrides = dict(timeout_overrides or {})
    self._startup_timeout = startup_timeout
    self._default_tool_timeout = default_tool_timeout

  async def startup(self) -> None:
    async with self._lock:
      if self._started:
        return

      if MCP_IMPORT_ERROR is not None:
        log.warning("MCP runtime unavailable; skipping startup: %s", MCP_IMPORT_ERROR)
        self._started = True
        return

      config = self._read_claude_config()
      mcp_servers = config.get("mcpServers", {})
      if not isinstance(mcp_servers, dict):
        mcp_servers = {}
      mcp_servers = dict(mcp_servers)
      mcp_servers.update(self._inline_servers)
      if not mcp_servers:
        self._started = True
        return

      connect_tasks = []
      for server_name, server_config in mcp_servers.items():
        if self._allowed_servers is not None and server_name not in self._allowed_servers:
          continue
        if not isinstance(server_config, dict):
          log.warning("Skipping MCP server %s: invalid config", server_name)
          continue

        server_type = str(server_config.get("type", "stdio")).strip().lower()
        if server_type != "stdio":
          log.info("Skipping MCP server %s: unsupported type %s", server_name, server_type)
          continue

        connect_tasks.append(self._connect_or_warn(server_name, server_config))

      if connect_tasks:
        for state in await asyncio.gather(*connect_tasks):
          if state is not None:
            self._servers[state.name] = state

      self._apply_collision_filtering()
      self._started = True

  async def _connect_or_warn(self, name: str, config: Dict[str, Any]) -> _ServerState | None:
    try:
      return await self._connect(name, config)
    except Exception as exc:
      log.warning("MCP server %s failed to connect: %s", name, exc)
      return None

  async def _connect(self, name: str, config: Dict[str, Any]) -> _ServerState:
    exit_contexts: List[Any] = []
    success = False
    try:
      command = str(config.get("command", "")).strip()
      if not command:
        raise ValueError("missing command")

      args_raw = config.get("args", [])
      args = [str(arg) for arg in args_raw] if isinstance(args_raw, list) else []

      env = dict(os.environ)
      env_raw = config.get("env")
      if isinstance(env_raw, dict):
        for key, value in env_raw.items():
          if value is None:
            continue
          env[str(key)] = str(value)

      cwd = config.get("cwd")
      server_params = StdioServerParameters(
        command=command,
        args=args,
        env=env,
        cwd=cwd,
      )

      devnull = open(os.devnull, "w")
      stdio_cm = stdio_client(server_params, errlog=devnull)
      read_stream, write_stream = await stdio_cm.__aenter__()
      exit_contexts.append(stdio_cm)

      session = ClientSession(read_stream, write_stream)
      await session.__aenter__()
      exit_contexts.append(session)

      await asyncio.wait_for(session.initialize(), timeout=self._startup_timeout)

      tools: List[Any] = []
      cursor: str | None = None
      while True:
        listed = await asyncio.wait_for(
          session.list_tools(cursor=cursor),
          timeout=self._startup_timeout,
        )
        tools.extend(listed.tools or [])
        if listed.nextCursor is None:
          break
        cursor = listed.nextCursor

      tool_definitions: List[Dict[str, Any]] = []
      for tool in tools:
        input_schema = tool.inputSchema or {"type": "object", "properties": {}}
        tool_definitions.append(
          {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": copy.deepcopy(input_schema),
          }
        )

      success = True
      return _ServerState(
        name=name,
        session=session,
        exit_contexts=exit_contexts,
        tool_definitions=tool_definitions,
        tool_names={tool["name"] for tool in tool_definitions},
      )
    finally:
      if not success:
        await self._close_contexts(exit_contexts)

  def get_tool_definitions(self) -> List[Dict[str, Any]]:
    return copy.deepcopy(self._tool_definitions)

  def get_server_tool_definitions(self, server_names: Set[str]) -> List[Dict[str, Any]]:
    tool_definitions: List[Dict[str, Any]] = []
    for server_name, state in self._servers.items():
      if server_name in server_names:
        tool_definitions.extend(copy.deepcopy(state.tool_definitions))
    return tool_definitions

  def get_server_names(self) -> Set[str]:
    return set(self._servers.keys())

  def get_server_catalog(self) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    for server_name, state in self._servers.items():
      tool_names = sorted(tool["name"] for tool in state.tool_definitions if isinstance(tool.get("name"), str))
      catalog[server_name] = {
        "tool_count": len(tool_names),
        "tools": tool_names,
      }
    return catalog

  def is_mcp_tool(self, name: str) -> bool:
    return name in self._mcp_tool_names

  def get_server_for_tool(self, name: str) -> str | None:
    return self._tool_to_server.get(name)

  async def call_tool(
    self,
    name: str,
    tool_input: Dict[str, Any],
  ) -> Tuple[Any | None, Dict[str, Any] | None]:
    server_name = self._tool_to_server.get(name)
    if not server_name:
      return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

    server = self._servers.get(server_name)
    if not server:
      return None, {"code": "mcp_tool_error", "message": f"MCP server unavailable: {server_name}"}

    timeout_seconds = self._timeout_overrides.get(server_name, self._default_tool_timeout)

    try:
      result = await server.session.call_tool(
        name,
        tool_input,
        read_timeout_seconds=timedelta(seconds=timeout_seconds),
      )
    except Exception as exc:
      msg = str(exc)
      return None, {
        "code": "tool_error",
        "sub_code": _classify_exception(exc, msg),
        "message": msg,
      }

    if result.isError:
      message = self._extract_text(result.content)
      if not message and result.structuredContent is not None:
        message = json.dumps(result.structuredContent, default=str)
      return None, {
        "code": "mcp_tool_error",
        "sub_code": _classify_mcp_error(message or ""),
        "message": message or f"MCP tool failed: {name}",
      }

    if result.structuredContent is not None:
      return result.structuredContent, None

    text_payload = self._extract_text(result.content)
    if text_payload:
      try:
        return json.loads(text_payload), None
      except json.JSONDecodeError:
        return {"text": text_payload}, None

    return {}, None

  async def shutdown(self) -> None:
    async with self._lock:
      if not self._started and not self._servers:
        return

      for server in reversed(list(self._servers.values())):
        await self._close_contexts(server.exit_contexts)

      self._servers.clear()
      self._tool_definitions = []
      self._tool_to_server = {}
      self._mcp_tool_names = set()
      self._started = False

  def _apply_collision_filtering(self) -> None:
    existing_names = set(self._builtin_tool_names)
    seen_mcp_names: Dict[str, str] = {}
    merged: List[Dict[str, Any]] = []

    self._tool_to_server = {}

    for server_name, state in self._servers.items():
      filtered: List[Dict[str, Any]] = []
      filtered_names: Set[str] = set()

      for tool in state.tool_definitions:
        tool_name = tool["name"]
        if tool_name in existing_names:
          log.warning("Skipping MCP tool %s from %s: collides with built-in tool", tool_name, server_name)
          continue

        first_server = seen_mcp_names.get(tool_name)
        if first_server:
          log.warning(
            "Skipping MCP tool %s from %s: collides with MCP tool from %s",
            tool_name,
            server_name,
            first_server,
          )
          continue

        seen_mcp_names[tool_name] = server_name
        self._tool_to_server[tool_name] = server_name
        filtered.append(tool)
        filtered_names.add(tool_name)

      state.tool_definitions = filtered
      state.tool_names = filtered_names
      merged.extend(filtered)

      log.info(
        "MCP server %s connected | %d tools: %s",
        server_name,
        len(filtered_names),
        sorted(filtered_names),
      )

    self._tool_definitions = merged
    self._mcp_tool_names = set(self._tool_to_server.keys())

  @staticmethod
  def _extract_text(content: Any) -> str:
    if not content:
      return ""

    chunks: List[str] = []
    for item in content:
      if isinstance(item, dict):
        text = item.get("text")
      else:
        text = getattr(item, "text", None)
      if isinstance(text, str) and text.strip():
        chunks.append(text)
    return "\n".join(chunks).strip()

  @staticmethod
  async def _close_contexts(contexts: List[Any]) -> None:
    while contexts:
      ctx = contexts.pop()
      try:
        await ctx.__aexit__(None, None, None)
      except Exception as exc:
        log.debug("MCP context close failed: %s", exc)

  def _read_claude_config(self) -> Dict[str, Any]:
    if self._config_path is None or not self._config_path.exists():
      return {}

    try:
      with open(self._config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    except Exception as exc:
      log.warning("Failed to read %s: %s", self._config_path, exc)
      return {}

    return data if isinstance(data, dict) else {}
