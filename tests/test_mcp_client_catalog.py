# ruff: noqa: E402

import asyncio
import builtins
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import policy_imports
import agent_gateway.mcp_client as mcp_client_module
from agent_gateway.mcp_client import McpClientManager, _ServerState
from agent_gateway.mcp_client_catalog import apply_collision_filtering
import api.agent.shared.server_policies as api_server_policies


class _CaptureLogger:
  def __init__(self) -> None:
    self.warnings: list[tuple[object, ...]] = []
    self.infos: list[tuple[object, ...]] = []
    self.errors: list[tuple[object, ...]] = []

  def warning(self, message, *args) -> None:
    self.warnings.append((message, *args))

  def info(self, message, *args) -> None:
    self.infos.append((message, *args))

  def error(self, message, *args) -> None:
    self.errors.append((message, *args))


def _run(coro):
  return asyncio.run(coro)


def _install_source_html_resolver(monkeypatch, resolver) -> None:
  research_module = ModuleType("research")
  source_html_module = ModuleType("research.source_html")
  source_html_module.sec_native_symbol_cached_only = resolver
  research_module.source_html = source_html_module
  monkeypatch.setitem(sys.modules, "research", research_module)
  monkeypatch.setitem(sys.modules, "research.source_html", source_html_module)


def _brk_resolver(value):
  return {
    "BRK-B": "BRK-B",
    "BRK.B": "BRK-B",
    "BRKB": "BRK-B",
  }.get(str(value))


def test_apply_collision_filtering_handles_collisions_prefixes_and_hidden_fields() -> None:
  logger = _CaptureLogger()
  first_tool = {
    "name": "shared_tool",
    "description": "first",
    "input_schema": {
      "type": "object",
      "properties": {"visible": {}, "_session_id": {}},
      "required": ["visible", "_session_id"],
    },
  }
  servers = {
    "first": SimpleNamespace(
      tool_prefix="",
      tool_definitions=[
        {"name": "builtin_tool", "description": "collision", "input_schema": {}},
        first_tool,
      ],
      tool_names=set(),
    ),
    "second": SimpleNamespace(
      tool_prefix="",
      tool_definitions=[{"name": "shared_tool", "description": "duplicate", "input_schema": {}}],
      tool_names=set(),
    ),
    "prefixed": SimpleNamespace(
      tool_prefix="safe_",
      tool_definitions=[{"name": "shared_tool", "description": "prefixed", "input_schema": {}}],
      tool_names=set(),
    ),
  }

  result = apply_collision_filtering(
    servers=servers,
    builtin_tool_names={"builtin_tool"},
    strip_input_fields={"_session_id"},
    logger=logger,
  )

  assert [tool["name"] for tool in result.tool_definitions] == ["shared_tool", "safe_shared_tool"]
  assert result.tool_to_server == {"shared_tool": "first", "safe_shared_tool": "prefixed"}
  assert result.prefixed_to_original == {"safe_shared_tool": "shared_tool"}
  assert result.mcp_tool_names == {"shared_tool", "safe_shared_tool"}
  assert servers["first"].tool_definitions == [first_tool]
  assert servers["first"].tool_names == {"shared_tool"}
  assert servers["second"].tool_definitions == []
  assert servers["second"].tool_names == set()
  assert servers["prefixed"].tool_definitions[0]["name"] == "safe_shared_tool"
  assert first_tool["input_schema"]["properties"] == {"visible": {}}
  assert first_tool["input_schema"]["required"] == ["visible"]
  assert len(logger.warnings) == 2
  assert logger.warnings[0][1:3] == ("builtin_tool", "first")
  assert logger.warnings[1][1:4] == ("shared_tool", "second", "first")
  assert len(logger.infos) == 3


def test_apply_collision_filtering_enriches_filing_table_tool_guidance() -> None:
  logger = _CaptureLogger()
  servers = {
    "edgar": SimpleNamespace(
      tool_prefix="edgar_",
      tool_definitions=[
        {
          "name": "search_filing_tables",
          "description": "Search filing table metadata.",
          "input_schema": {
            "type": "object",
            "properties": {
              "description": {"type": "string", "description": "Search text."},
              "period_from": {"type": "string"},
              "period_to": {"type": "string"},
              "section_key": {"type": "string"},
            },
          },
        },
        {
          "name": "get_filing_tables",
          "description": "Read filing tables.",
          "input_schema": {
            "type": "object",
            "properties": {
              "ticker": {"type": "string"},
              "year": {"type": "integer", "description": "Year."},
              "quarter": {"type": "integer"},
              "section": {"type": "string"},
              "table_id": {"type": "string"},
            },
          },
        },
      ],
      tool_names=set(),
    ),
  }

  result = apply_collision_filtering(
    servers=servers,
    builtin_tool_names=set(),
    strip_input_fields=set(),
    logger=logger,
  )

  by_name = {tool["name"]: tool for tool in result.tool_definitions}
  search_tool = by_name["edgar_search_filing_tables"]
  get_tool = by_name["edgar_get_filing_tables"]
  assert "Do not pass query, text, year, quarter" in search_tool["description"]
  assert "returned table_id" in search_tool["description"]
  assert "Use this instead of query or text" in search_tool["input_schema"]["properties"]["description"]["description"]
  assert "Do not use fiscal_period or period" in search_tool["input_schema"]["properties"]["period_from"]["description"]
  assert "Do not pass form_type, limit, query" in get_tool["description"]
  assert "sections is invalid, use section" in get_tool["description"]
  assert "Do not pass strings like '2024-FY'" in get_tool["input_schema"]["properties"]["year"]["description"]
  assert "Do not pass string values" in get_tool["input_schema"]["properties"]["quarter"]["description"]
  assert result.prefixed_to_original == {
    "edgar_search_filing_tables": "search_filing_tables",
    "edgar_get_filing_tables": "get_filing_tables",
  }


def test_apply_collision_filtering_enriches_filing_read_and_get_filings_guidance() -> None:
  logger = _CaptureLogger()
  servers = {
    "research-corpus": SimpleNamespace(
      tool_prefix="",
      tool_definitions=[
        {
          "name": "filings_read",
          "description": "Read filing excerpts.",
          "input_schema": {
            "type": "object",
            "properties": {
              "document_id": {"type": "string"},
              "section": {"type": "string"},
              "char_start": {"type": "integer"},
              "char_end": {"type": "integer"},
              "offset_frame": {"type": "string"},
            },
          },
        },
      ],
      tool_names=set(),
    ),
    "edgar-parser": SimpleNamespace(
      tool_prefix="edgar_",
      tool_definitions=[
        {
          "name": "get_filings",
          "description": "Get filings.",
          "input_schema": {
            "type": "object",
            "properties": {
              "ticker": {"type": "string"},
              "year": {"type": "integer"},
              "quarter": {"type": "integer"},
              "source": {"type": "string"},
            },
          },
        },
      ],
      tool_names=set(),
    ),
  }

  result = apply_collision_filtering(
    servers=servers,
    builtin_tool_names=set(),
    strip_input_fields=set(),
    logger=logger,
  )

  by_name = {tool["name"]: tool for tool in result.tool_definitions}
  filings_read = by_name["filings_read"]
  get_filings = by_name["edgar_get_filings"]
  assert "Copy document_id and section from filings_search or filings_list results" in filings_read["description"]
  assert "copy char_start/char_end from filings_search hits and set offset_frame='document'" in (
    filings_read["description"]
  )
  assert "Do not invent or normalize section names" in filings_read["description"]
  assert "use transcripts_read for transcript results" in filings_read["description"]
  assert "Exact document_id copied from filings_search or filings_list" in (
    filings_read["input_schema"]["properties"]["document_id"]["description"]
  )
  assert "Transcript ids such as fmp_transcripts:* belong in transcripts_read" in (
    filings_read["input_schema"]["properties"]["document_id"]["description"]
  )
  assert "Do not invent normalized names" in filings_read["input_schema"]["properties"]["section"]["description"]
  assert "Use 'document' for offsets from search hits" in (
    filings_read["input_schema"]["properties"]["offset_frame"]["description"]
  )
  assert "not a broad filing-list endpoint" in get_filings["description"]
  assert "fiscal/reporting integer year" in get_filings["description"]
  assert "optional source" in get_filings["description"]
  assert "Do not pass form_type" in get_filings["description"]
  assert "Do not use this tool for broad discovery" in get_filings["input_schema"]["properties"]["ticker"]["description"]
  assert "Required integer fiscal/reporting year" in get_filings["input_schema"]["properties"]["year"]["description"]
  assert "not 'FY'" in get_filings["input_schema"]["properties"]["quarter"]["description"]
  assert "Optional supported filing source/mode" in get_filings["input_schema"]["properties"]["source"]["description"]
  assert result.prefixed_to_original == {"edgar_get_filings": "get_filings"}


def test_apply_collision_filtering_enriches_price_target_identity_guidance() -> None:
  logger = _CaptureLogger()
  servers = {
    "portfolio-reads": SimpleNamespace(
      tool_prefix="portfolio_",
      tool_definitions=[
        {
          "name": "get_price_target",
          "description": "Read a price target.",
          "input_schema": {
            "type": "object",
            "properties": {
              "research_file_id": {"type": "integer", "description": "Research file id."},
              "handoff_id": {"type": "integer"},
              "format": {"type": "string"},
            },
            "required": ["research_file_id"],
          },
        },
      ],
      tool_names=set(),
    ),
  }

  result = apply_collision_filtering(
    servers=servers,
    builtin_tool_names=set(),
    strip_input_fields=set(),
    logger=logger,
  )

  price_target = result.tool_definitions[0]
  assert price_target["name"] == "portfolio_get_price_target"
  assert "Pass research_file_id" in price_target["description"]
  assert "Do not pass ticker" in price_target["description"]
  assert "first resolve the typed research file" in price_target["description"]
  assert "sell-side analyst consensus" in price_target["description"]
  props = price_target["input_schema"]["properties"]
  assert "This is not a ticker" in props["research_file_id"]["description"]
  assert "Resolve it from thesis_list, list_research_files" in props["research_file_id"]["description"]
  assert "omit for the current model handoff" in props["handoff_id"]["description"]
  assert "does not change the required research_file_id identity" in props["format"]["description"]


def test_apply_collision_filtering_enriches_model_insights_identity_guidance() -> None:
  logger = _CaptureLogger()
  servers = {
    "portfolio-reads": SimpleNamespace(
      tool_prefix="portfolio_",
      tool_definitions=[
        {
          "name": "get_model_insights",
          "description": "Read model insights.",
          "input_schema": {
            "type": "object",
            "properties": {
              "research_file_id": {"type": "integer", "description": "Research file id."},
              "model_insights_id": {"type": "string"},
              "format": {"type": "string"},
            },
            "required": ["research_file_id"],
          },
        },
      ],
      tool_names=set(),
    ),
  }

  result = apply_collision_filtering(
    servers=servers,
    builtin_tool_names=set(),
    strip_input_fields=set(),
    logger=logger,
  )

  insights = result.tool_definitions[0]
  assert insights["name"] == "portfolio_get_model_insights"
  assert "Pass research_file_id" in insights["description"]
  assert "Do not pass ticker or query" in insights["description"]
  assert "first resolve the typed research file" in insights["description"]
  props = insights["input_schema"]["properties"]
  assert "This is not a ticker" in props["research_file_id"]["description"]
  assert "Resolve it from thesis_list, list_research_files" in props["research_file_id"]["description"]
  assert "omit to read the latest current-model insights" in props["model_insights_id"]["description"]
  assert "does not change the required research_file_id identity" in props["format"]["description"]


def test_apply_collision_filtering_enriches_industry_peer_comparison_symbol_guidance() -> None:
  logger = _CaptureLogger()
  servers = {
    "portfolio-reads": SimpleNamespace(
      tool_prefix="portfolio_",
      tool_definitions=[
        {
          "name": "industry_peer_comparison",
          "description": "Compare peers.",
          "input_schema": {
            "type": "object",
            "properties": {
              "symbol": {"type": "string", "description": "Company symbol."},
            },
            "required": ["symbol"],
          },
        },
      ],
      tool_names=set(),
    ),
  }

  result = apply_collision_filtering(
    servers=servers,
    builtin_tool_names=set(),
    strip_input_fields=set(),
    logger=logger,
  )

  comparison = result.tool_definitions[0]
  assert comparison["name"] == "portfolio_industry_peer_comparison"
  assert "Pass symbol" in comparison["description"]
  assert "Do not pass ticker" in comparison["description"]
  assert "compare_peers" in comparison["description"]
  props = comparison["input_schema"]["properties"]
  assert "Do not pass ticker" in props["symbol"]["description"]
  assert "use symbol" in props["symbol"]["description"]


def test_apply_collision_filtering_enriches_gsheets_canonical_argument_guidance() -> None:
  logger = _CaptureLogger()
  tool_properties = {
    "gsheets_list_tabs": ["spreadsheet"],
    "gsheets_read_range": [
      "spreadsheet",
      "range",
      "value_render_option",
      "date_time_render_option",
    ],
    "gsheets_write_range": ["spreadsheet", "range", "values"],
    "gsheets_append_rows": ["spreadsheet", "range", "values"],
    "gsheets_clear_range": ["spreadsheet", "range"],
    "gsheets_recalculate_range": ["spreadsheet", "range"],
    "gsheets_create_spreadsheet": ["title"],
    "gsheets_copy_spreadsheet": ["spreadsheet", "title", "tabs"],
    "gsheets_search_spreadsheets": ["query", "limit"],
  }
  servers = {
    "gsheets-mcp": SimpleNamespace(
      tool_prefix="",
      tool_definitions=[
        {
          "name": tool_name,
          "description": f"Original description for {tool_name}.",
          "input_schema": {
            "type": "object",
            "properties": {
              property_name: {"type": "string", "description": "Original property description."}
              for property_name in property_names
            },
          },
        }
        for tool_name, property_names in tool_properties.items()
      ],
      tool_names=set(),
    ),
  }

  result = apply_collision_filtering(
    servers=servers,
    builtin_tool_names=set(),
    strip_input_fields=set(),
    logger=logger,
  )

  by_name = {tool["name"]: tool for tool in result.tool_definitions}
  assert set(by_name) == set(tool_properties)
  for tool_name, tool in by_name.items():
    assert tool["description"].startswith(f"Original description for {tool_name}.")
    for prop in tool["input_schema"]["properties"].values():
      assert prop["description"] == "Original property description."

  assert "returned normalized spreadsheet" in by_name["gsheets_list_tabs"]["description"]
  assert "Reuse the returned spreadsheet and range" in by_name["gsheets_read_range"]["description"]
  assert "do not blindly replay it" in by_name["gsheets_write_range"]["description"]
  assert "Append is non-idempotent" in by_name["gsheets_append_rows"]["description"]
  assert "Creation is non-idempotent" in by_name["gsheets_create_spreadsheet"]["description"]
  assert "destination recovery details" in by_name["gsheets_copy_spreadsheet"]["description"]
  assert "local mode only" in by_name["gsheets_search_spreadsheets"]["description"]
  assert "read the target range" in by_name["gsheets_clear_range"]["description"]
  assert "formula restoration" in by_name["gsheets_recalculate_range"]["description"]

  rendered = "\n".join(tool["description"] for tool in by_name.values())
  for legacy_term in ("cell_range", "spreadsheet_id", "gsheet_id", "new_title"):
    assert legacy_term not in rendered


def test_apply_collision_filtering_enriches_filing_document_event_and_transcript_guidance() -> None:
  logger = _CaptureLogger()
  servers = {
    "research-corpus": SimpleNamespace(
      tool_prefix="",
      tool_definitions=[
        {
          "name": "transcripts_read",
          "description": "Read transcript excerpts.",
          "input_schema": {
            "type": "object",
            "properties": {
              "document_id": {"type": "string"},
              "section": {"type": "string"},
              "char_start": {"type": "integer"},
              "char_end": {"type": "integer"},
              "offset_frame": {"type": "string"},
            },
          },
        },
      ],
      tool_names=set(),
    ),
    "edgar-parser": SimpleNamespace(
      tool_prefix="edgar_",
      tool_definitions=[
        {
          "name": "get_filing_document",
          "description": "Read filing document.",
          "input_schema": {
            "type": "object",
            "properties": {
              "ticker": {"type": "string"},
              "accession": {"type": "string"},
              "section": {"type": "string"},
              "sections": {"type": "array"},
            },
          },
        },
        {
          "name": "get_event_filings",
          "description": "Get event filings.",
          "input_schema": {
            "type": "object",
            "properties": {
              "ticker": {"type": "string"},
              "form_type": {"type": "string"},
            },
          },
        },
      ],
      tool_names=set(),
    ),
  }

  result = apply_collision_filtering(
    servers=servers,
    builtin_tool_names=set(),
    strip_input_fields=set(),
    logger=logger,
  )

  by_name = {tool["name"]: tool for tool in result.tool_definitions}
  transcripts_read = by_name["transcripts_read"]
  filing_document = by_name["edgar_get_filing_document"]
  event_filings = by_name["edgar_get_event_filings"]
  assert "only supported section values are prepared_remarks and qa" in transcripts_read["description"]
  assert "Do not use filings_read for transcript document_ids" in transcripts_read["description"]
  assert "fmp_transcripts:" in transcripts_read["input_schema"]["properties"]["document_id"]["description"]
  assert "prepared_remarks or qa only" in transcripts_read["input_schema"]["properties"]["section"]["description"]
  assert "Do not pass corpus document_id values" in filing_document["description"]
  assert "first call get_filing_sections and copy the exact returned section key/name" in (
    filing_document["description"]
  )
  assert "use search_filing_text" in filing_document["description"]
  assert "use filings_read with a document_id and offsets copied from corpus results" in (
    filing_document["description"]
  )
  assert "Exact section key/name returned by get_filing_sections" in (
    filing_document["input_schema"]["properties"]["section"]["description"]
  )
  assert "Do not pass topical labels or normalized aliases" in (
    filing_document["input_schema"]["properties"]["sections"]["description"]
  )
  assert "Do not pass raw edgar: document_id strings" in (
    filing_document["input_schema"]["properties"]["accession"]["description"]
  )
  assert "Do not request periodic 10-K or 10-Q filings here" in event_filings["description"]
  assert "Do not pass 10-K or 10-Q" in event_filings["input_schema"]["properties"]["form_type"]["description"]
  assert result.prefixed_to_original == {
    "edgar_get_filing_document": "get_filing_document",
    "edgar_get_event_filings": "get_event_filings",
  }


def test_parent_apply_collision_filtering_uses_parent_logger(monkeypatch) -> None:
  logger = _CaptureLogger()
  monkeypatch.setattr(mcp_client_module, "log", logger)
  manager = McpClientManager(config_path=None, builtin_tool_names={"builtin_tool"})
  manager._servers = {
    "gateway": _ServerState(
      name="gateway",
      session=object(),
      exit_contexts=[],
      tool_definitions=[
        {"name": "builtin_tool", "description": "collision", "input_schema": {}},
        {"name": "remote_tool", "description": "kept", "input_schema": {}},
      ],
      tool_names={"builtin_tool", "remote_tool"},
    ),
    "prefixed": _ServerState(
      name="prefixed",
      session=object(),
      exit_contexts=[],
      tool_definitions=[
        {"name": "remote_tool", "description": "prefixed", "input_schema": {}},
      ],
      tool_names={"remote_tool"},
      tool_prefix="safe_",
    ),
  }

  manager._apply_collision_filtering()

  assert manager.get_tool_definitions() == [
    {"name": "remote_tool", "description": "kept", "input_schema": {}},
    {"name": "safe_remote_tool", "description": "prefixed", "input_schema": {}},
  ]
  assert manager.is_mcp_tool("remote_tool") is True
  assert manager.is_mcp_tool("safe_remote_tool") is True
  assert manager.get_server_for_tool("remote_tool") == "gateway"
  assert manager.get_server_for_tool("safe_remote_tool") == "prefixed"
  assert manager.resolve_tool_name("prefixed", "remote_tool") == "safe_remote_tool"
  assert logger.warnings[0][1:3] == ("builtin_tool", "gateway")
  assert logger.infos[0][1:4] == ("gateway", 1, ["remote_tool"])
  assert logger.infos[1][1:4] == ("prefixed", 1, ["safe_remote_tool"])


def test_policy_owner_prefilter_hides_split_server_duplicates_before_collision_warning(monkeypatch) -> None:
  logger = _CaptureLogger()
  monkeypatch.setattr(mcp_client_module, "log", logger)
  manager = McpClientManager(config_path=None)
  manager._servers = {
    "portfolio-reads-mcp": _ServerState(
      name="portfolio-reads-mcp",
      session=object(),
      exit_contexts=[],
      tool_definitions=[
        {"name": "get_mcp_context", "description": "reads owner", "input_schema": {}},
      ],
      tool_names={"get_mcp_context"},
    ),
    "portfolio-writes-mcp": _ServerState(
      name="portfolio-writes-mcp",
      session=object(),
      exit_contexts=[],
      tool_definitions=[
        {"name": "get_mcp_context", "description": "stale residual duplicate", "input_schema": {}},
        {"name": "apply_patch_ops", "description": "residual writer", "input_schema": {}},
      ],
      tool_names={"get_mcp_context", "apply_patch_ops"},
    ),
  }

  manager._apply_collision_filtering(
    policy_server_for_tool=lambda tool_name: (
      "portfolio-reads-mcp" if tool_name == "get_mcp_context" else "portfolio-writes-mcp"
    )
  )

  assert manager.get_tool_definitions() == [
    {"name": "get_mcp_context", "description": "reads owner", "input_schema": {}},
    {"name": "apply_patch_ops", "description": "residual writer", "input_schema": {}},
  ]
  assert manager.get_server_for_tool("get_mcp_context") == "portfolio-reads-mcp"
  assert manager.get_server_for_tool("apply_patch_ops") == "portfolio-writes-mcp"
  assert manager._servers["portfolio-writes-mcp"].tool_names == {"apply_patch_ops"}
  assert "get_mcp_context" in manager.get_startup_diagnostics()["portfolio-writes-mcp"]["message"]
  assert not any("collides with MCP tool" in str(warning[0]) for warning in logger.warnings)


def test_policy_owner_mismatch_hides_residual_runtime_tool(monkeypatch) -> None:
  logger = _CaptureLogger()
  monkeypatch.setattr(mcp_client_module, "log", logger)
  manager = McpClientManager(config_path=None)
  manager._servers = {
    "portfolio-reads-mcp": _ServerState(
      name="portfolio-reads-mcp",
      session=object(),
      exit_contexts=[],
      tool_definitions=[
        {"name": "execute_trade", "description": "stale residual", "input_schema": {}},
        {"name": "preview_trade", "description": "current residual", "input_schema": {}},
      ],
      tool_names={"execute_trade", "preview_trade"},
    ),
  }

  manager._apply_collision_filtering(
    policy_server_for_tool=lambda tool_name: (
      "portfolio-trades-mcp" if tool_name == "execute_trade" else "portfolio-reads-mcp"
    )
  )

  assert manager.get_tool_definitions() == [
    {"name": "preview_trade", "description": "current residual", "input_schema": {}},
  ]
  assert manager.is_mcp_tool("execute_trade") is False
  assert manager.is_mcp_tool("preview_trade") is True
  assert manager.get_server_for_tool("execute_trade") is None
  assert manager.get_server_for_tool("preview_trade") == "portfolio-reads-mcp"
  assert manager._servers["portfolio-reads-mcp"].tool_names == {"preview_trade"}
  assert manager.get_startup_diagnostics()["portfolio-reads-mcp"]["category"] == "policy_owner_mismatch"
  assert "execute_trade" in manager.get_startup_diagnostics()["portfolio-reads-mcp"]["message"]
  assert logger.errors


def test_strict_runtime_tool_set_hides_unclassified_gsheets_tools(monkeypatch) -> None:
  logger = _CaptureLogger()
  monkeypatch.setattr(mcp_client_module, "log", logger)
  manager = McpClientManager(config_path=None)
  manager._servers = {
    "gsheets-mcp": _ServerState(
      name="gsheets-mcp",
      session=object(),
      exit_contexts=[],
      tool_definitions=[
        {"name": "gsheets_read_range", "description": "broker read", "input_schema": {}},
        {"name": "gsheets_search_spreadsheets", "description": "local only", "input_schema": {}},
        {"name": "gsheet_read_range", "description": "removed legacy name", "input_schema": {}},
      ],
      tool_names={"gsheets_read_range", "gsheets_search_spreadsheets", "gsheet_read_range"},
    ),
  }

  manager._apply_collision_filtering()

  assert [tool["name"] for tool in manager.get_tool_definitions()] == ["gsheets_read_range"]
  assert manager.get_server_for_tool("gsheets_read_range") == "gsheets-mcp"
  assert manager.get_server_for_tool("gsheets_search_spreadsheets") is None
  assert manager.get_server_for_tool("gsheet_read_range") is None
  assert manager._servers["gsheets-mcp"].tool_names == {"gsheets_read_range"}
  diagnostic = manager.get_startup_diagnostics()["gsheets-mcp"]
  assert diagnostic["category"] == "strict_runtime_tool_set_mismatch"
  assert "gsheets_search_spreadsheets->unclassified" in diagnostic["message"]
  assert "gsheet_read_range->unclassified" in diagnostic["message"]


def test_gsheets_closed_world_survives_shared_policy_import_failure(monkeypatch) -> None:
  logger = _CaptureLogger()
  monkeypatch.setattr(mcp_client_module, "log", logger)
  monkeypatch.setattr(
    mcp_client_module,
    "load_server_policy_helpers",
    lambda: (None, None, None),
  )
  monkeypatch.setattr(mcp_client_module, "load_server_policy_module", lambda: None)
  manager = McpClientManager(config_path=None)
  manager._servers = {
    "gsheets-mcp": _ServerState(
      name="gsheets-mcp",
      session=object(),
      exit_contexts=[],
      tool_definitions=[
        {"name": "gsheets_read_range", "description": "broker read", "input_schema": {}},
        {"name": "gsheets_search_spreadsheets", "description": "local only", "input_schema": {}},
        {"name": "gsheet_read_range", "description": "removed legacy name", "input_schema": {}},
      ],
      tool_names={"gsheets_read_range", "gsheets_search_spreadsheets", "gsheet_read_range"},
    ),
  }

  manager._apply_collision_filtering()

  assert [tool["name"] for tool in manager.get_tool_definitions()] == ["gsheets_read_range"]
  assert manager.get_server_for_tool("gsheets_read_range") == "gsheets-mcp"
  assert manager.get_server_for_tool("gsheets_search_spreadsheets") is None
  assert manager.get_server_for_tool("gsheet_read_range") is None
  diagnostic = manager.get_startup_diagnostics()["gsheets-mcp"]
  assert diagnostic["category"] == "strict_runtime_tool_set_mismatch"
  assert any("built-in Google Sheets cutover policy" in str(entry[0]) for entry in logger.warnings)


def test_gsheets_read_classification_survives_shared_policy_import_failure(monkeypatch) -> None:
  monkeypatch.setattr(
    mcp_client_module,
    "load_server_policy_helpers",
    lambda: (None, None, None),
  )
  manager = McpClientManager(config_path=None)
  manager._servers = {
    "gsheets-mcp": _ServerState(
      name="gsheets-mcp",
      session=object(),
      exit_contexts=[object()],
      tool_definitions=[],
      tool_names={"gsheets_read_range"},
      config={"type": "stdio", "command": "gsheets-mcp"},
    ),
  }
  manager._tool_to_server = {"gsheets_read_range": "gsheets-mcp"}
  reconnects = []

  async def time_out(**_kwargs):
    raise asyncio.TimeoutError("timeout with upstream detail")

  async def reconnect_only_for_future(**kwargs):
    reconnects.append(kwargs["original_name"])
    return True

  manager._call_tool_once = time_out
  manager._reconnect_stdio_server_for_future = reconnect_only_for_future

  result, error = _run(manager.call_tool("gsheets_read_range", {}))

  assert result is None
  assert error["sub_code"] == "sheets_transport_error"
  assert error["data"]["error"]["outcome"] == {
    "state": "unchanged",
    "phase": "dispatch",
    "mutation_may_have_occurred": False,
  }
  assert error["data"]["error"]["retry"]["safe"] is True
  assert error["data"]["error"]["retry"]["automatic"] is False
  assert "upstream detail" not in str(error)
  assert reconnects == ["gsheets_read_range"]


def test_policy_owner_invariant_falls_back_to_api_import(monkeypatch) -> None:
  logger = _CaptureLogger()
  monkeypatch.setattr(mcp_client_module, "log", logger)

  def fake_import_module(name: str):
    if name == "agent.shared.server_policies":
      raise ModuleNotFoundError("No module named 'agent'", name="agent")
    if name == "api.agent.shared.server_policies":
      return api_server_policies
    raise AssertionError(f"unexpected import: {name}")

  monkeypatch.setattr(policy_imports.importlib, "import_module", fake_import_module)
  monkeypatch.setattr(
    api_server_policies,
    "get_server_for_policy_tool",
    lambda tool_name: "portfolio-trades-mcp" if tool_name == "execute_trade" else None,
  )
  manager = McpClientManager(config_path=None)
  manager._servers = {
    "portfolio-reads-mcp": _ServerState(
      name="portfolio-reads-mcp",
      session=object(),
      exit_contexts=[],
      tool_definitions=[
        {"name": "execute_trade", "description": "stale residual", "input_schema": {}},
      ],
      tool_names={"execute_trade"},
    ),
  }

  manager._apply_collision_filtering()

  assert manager.get_tool_definitions() == []
  assert manager.is_mcp_tool("execute_trade") is False
  assert manager.get_startup_diagnostics()["portfolio-reads-mcp"]["category"] == "policy_owner_mismatch"


def test_policy_owner_invariant_raises_when_policy_import_dependency_breaks(
  monkeypatch,
) -> None:
  def fake_import_module(_name: str):
    raise ModuleNotFoundError("No module named 'broken_dependency'", name="broken_dependency")

  monkeypatch.setattr(policy_imports.importlib, "import_module", fake_import_module)
  manager = McpClientManager(config_path=None)
  manager._servers = {
    "portfolio-reads-mcp": _ServerState(
      name="portfolio-reads-mcp",
      session=object(),
      exit_contexts=[],
      tool_definitions=[
        {"name": "execute_trade", "description": "stale residual", "input_schema": {}},
      ],
      tool_names={"execute_trade"},
    ),
  }

  with pytest.raises(ModuleNotFoundError, match="broken_dependency"):
    manager._apply_collision_filtering()


def test_policy_owner_invariant_raises_when_api_policy_import_dependency_breaks(
  monkeypatch,
) -> None:
  def fake_import_module(name: str):
    if name == "agent.shared.server_policies":
      raise ModuleNotFoundError("No module named 'agent'", name="agent")
    if name == "api.agent.shared.server_policies":
      raise ModuleNotFoundError("No module named 'broken_dependency'", name="broken_dependency")
    raise AssertionError(f"unexpected import: {name}")

  monkeypatch.setattr(policy_imports.importlib, "import_module", fake_import_module)
  manager = McpClientManager(config_path=None)
  manager._servers = {
    "portfolio-reads-mcp": _ServerState(
      name="portfolio-reads-mcp",
      session=object(),
      exit_contexts=[],
      tool_definitions=[
        {"name": "execute_trade", "description": "stale residual", "input_schema": {}},
      ],
      tool_names={"execute_trade"},
    ),
  }

  with pytest.raises(ModuleNotFoundError, match="broken_dependency"):
    manager._apply_collision_filtering()


def test_policy_owner_invariant_uses_original_name_for_prefixed_tools(monkeypatch) -> None:
  logger = _CaptureLogger()
  monkeypatch.setattr(mcp_client_module, "log", logger)
  manager = McpClientManager(config_path=None)
  manager._servers = {
    "portfolio-trades-mcp": _ServerState(
      name="portfolio-trades-mcp",
      session=object(),
      exit_contexts=[],
      tool_definitions=[
        {"name": "execute_trade", "description": "split", "input_schema": {}},
      ],
      tool_names={"execute_trade"},
      tool_prefix="trades_",
    ),
  }

  manager._apply_collision_filtering(
    policy_server_for_tool=lambda tool_name: (
      "portfolio-trades-mcp" if tool_name == "execute_trade" else None
    )
  )

  assert manager.get_tool_definitions() == [
    {"name": "trades_execute_trade", "description": "split", "input_schema": {}},
  ]
  assert manager.get_server_for_tool("trades_execute_trade") == "portfolio-trades-mcp"
  assert manager.get_original_tool_name("trades_execute_trade") == "execute_trade"
  assert manager.get_original_tool_name("unknown_tool") == "unknown_tool"
  assert manager.resolve_tool_name("portfolio-trades-mcp", "execute_trade") == "trades_execute_trade"
  assert manager.get_startup_diagnostics() == {}
  assert logger.errors == []


def test_provider_symbol_translation_allows_scalar_symbol_and_ticker_keys(monkeypatch) -> None:
  _install_source_html_resolver(monkeypatch, _brk_resolver)
  manager = McpClientManager(config_path=None)

  assert manager._translate_provider_symbol("fetch_financials", {"symbol": "BRKB"}) == {"symbol": "BRK-B"}
  assert manager._translate_provider_symbol("get_filings", {"ticker": "BRKB"}) == {"ticker": "BRK-B"}


def test_provider_symbol_translation_leaves_non_allowlisted_tool_untouched(monkeypatch) -> None:
  _install_source_html_resolver(monkeypatch, _brk_resolver)
  manager = McpClientManager(config_path=None)
  payload = {"symbol": "BRKB"}

  assert manager._translate_provider_symbol("not_a_provider_tool", payload) is payload
  assert payload == {"symbol": "BRKB"}


def test_provider_symbol_translation_uses_original_name_for_prefixed_tool(monkeypatch) -> None:
  _install_source_html_resolver(monkeypatch, _brk_resolver)
  manager = McpClientManager(config_path=None)

  class _Session:
    def __init__(self) -> None:
      self.calls = []

    async def call_tool(self, name, tool_input, *, read_timeout_seconds, meta=None):
      self.calls.append(
        {
          "name": name,
          "tool_input": tool_input,
          "read_timeout_seconds": read_timeout_seconds,
          "meta": meta,
        }
      )
      return SimpleNamespace(isError=False, structuredContent={"ok": True}, content=None)

  session = _Session()
  manager._tool_to_server = {"safe_get_filings": "edgar-parser-mcp"}
  manager._prefixed_to_original = {"safe_get_filings": "get_filings"}
  manager._servers = {"edgar-parser-mcp": SimpleNamespace(session=session)}

  result, error = _run(manager.call_tool("safe_get_filings", {"ticker": "BRKB"}))

  assert error is None
  assert result == {"ok": True}
  assert session.calls[0]["name"] == "get_filings"
  assert session.calls[0]["tool_input"] == {"ticker": "BRK-B"}


def test_provider_symbol_translation_fmp_profile_dual_key_atomicity(monkeypatch) -> None:
  _install_source_html_resolver(monkeypatch, _brk_resolver)
  manager = McpClientManager(config_path=None)

  assert manager._translate_provider_symbol(
    "fetch_company_profile",
    {"symbol": "BRKB", "ticker": "BRK-B"},
  ) == {"symbol": "BRK-B", "ticker": "BRK-B"}

  conflicting = {"symbol": "BRKB", "ticker": "AAPL"}
  assert manager._translate_provider_symbol("fetch_company_profile", conflicting) == conflicting


def test_provider_symbol_translation_comma_tokens(monkeypatch) -> None:
  _install_source_html_resolver(monkeypatch, _brk_resolver)
  manager = McpClientManager(config_path=None)

  assert manager._translate_provider_symbol("get_news", {"symbols": "BRKB,AAPL"}) == {
    "symbols": "BRK-B,AAPL",
  }


def test_provider_symbol_translation_excludes_list_nested_and_secondary_fields(monkeypatch) -> None:
  _install_source_html_resolver(monkeypatch, _brk_resolver)
  manager = McpClientManager(config_path=None)

  compare_concept = {"tickers": ["BRKB", "AAPL"]}
  compare_filing_tables = {"tickers": ["BRKB", "AAPL"]}
  warm_metric_cache = {"items": [{"ticker": "BRKB"}]}
  assert manager._translate_provider_symbol("compare_concept", compare_concept) is compare_concept
  assert manager._translate_provider_symbol("compare_filing_tables", compare_filing_tables) is compare_filing_tables
  assert manager._translate_provider_symbol("warm_metric_cache", warm_metric_cache) is warm_metric_cache

  compare_peers = {"symbol": "BRKB", "peers": ["BRKB", "AAPL"]}
  assert manager._translate_provider_symbol("compare_peers", compare_peers) == {
    "symbol": "BRK-B",
    "peers": ["BRKB", "AAPL"],
  }
  assert manager._translate_provider_symbol("industry_peer_comparison", {"symbol": "BRKB"}) == {
    "symbol": "BRK-B",
  }

  event_filings = {"ticker": "BRKB", "related_tickers": ["BRKB", "AAPL"]}
  assert manager._translate_provider_symbol("get_event_filings", event_filings) == {
    "ticker": "BRK-B",
    "related_tickers": ["BRKB", "AAPL"],
  }

  filing_evidence = {"ticker": "BRKB", "related_tickers": ["BRKB", "AAPL"]}
  assert manager._translate_provider_symbol("get_filing_evidence", filing_evidence) == {
    "ticker": "BRK-B",
    "related_tickers": ["BRKB", "AAPL"],
  }


def test_provider_symbol_translation_copy_not_mutate_and_retry_reuses_effective_input(monkeypatch) -> None:
  _install_source_html_resolver(monkeypatch, _brk_resolver)
  manager = McpClientManager(
    config_path=None,
    logical_server_routes={"market-data-mcp": "fmp-mcp"},
  )
  manager._tool_to_server = {"fetch_financials": "market-data-mcp"}
  manager._dispatch_to_original = {"fetch_financials": "fmp_fetch"}
  manager._servers = {"fmp-mcp": SimpleNamespace(session=object())}
  original_input = {"symbol": "BRKB", "other": {"nested": True}}
  seen_inputs = []

  async def fake_call_tool_once(**kwargs):
    seen_inputs.append(kwargs["tool_input"])
    raise RuntimeError("closed transport")

  async def fake_retry_stdio_tool_call_after_reconnect(**kwargs):
    seen_inputs.append(kwargs["tool_input"])
    return SimpleNamespace(isError=False, structuredContent={"ok": True}, content=None)

  manager._call_tool_once = fake_call_tool_once
  manager._retry_stdio_tool_call_after_reconnect = fake_retry_stdio_tool_call_after_reconnect

  result, error = _run(manager.call_tool("fetch_financials", original_input))

  assert error is None
  assert result == {"ok": True}
  assert original_input == {"symbol": "BRKB", "other": {"nested": True}}
  assert seen_inputs[0] is seen_inputs[1]
  assert seen_inputs[0] is not original_input
  assert seen_inputs[0] == {"symbol": "BRK-B", "other": {"nested": True}}


def test_provider_symbol_translation_import_failure_returns_original(monkeypatch) -> None:
  manager = McpClientManager(config_path=None)
  payload = {"symbol": "BRKB"}
  real_import = builtins.__import__

  def fail_research_source_html_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "research.source_html":
      raise ImportError("blocked")
    return real_import(name, globals, locals, fromlist, level)

  monkeypatch.setattr(builtins, "__import__", fail_research_source_html_import)

  assert manager._translate_provider_symbol("fetch_financials", payload) is payload


def test_provider_symbol_translation_resolver_error_returns_original(monkeypatch) -> None:
  def raising_resolver(_value):
    raise RuntimeError("resolver failed")

  _install_source_html_resolver(monkeypatch, raising_resolver)
  manager = McpClientManager(config_path=None)
  payload = {"symbol": "BRKB"}

  assert manager._translate_provider_symbol("fetch_financials", payload) is payload


def test_generic_stdio_sheets_mutation_transport_loss_reconnects_without_replay() -> None:
  manager = McpClientManager(config_path=None)
  server = _ServerState(
    name="gsheets-mcp",
    session=object(),
    exit_contexts=[object()],
    tool_definitions=[],
    tool_names={"gsheets_write_range"},
    config={"type": "stdio", "command": "gsheets-mcp"},
  )
  manager._servers = {"gsheets-mcp": server}
  manager._tool_to_server = {"gsheets_write_range": "gsheets-mcp"}
  dispatches = []
  reconnects = []

  async def fail_once(**kwargs):
    dispatches.append(kwargs["original_name"])
    raise EOFError("lost transport with upstream detail")

  async def reconnect_only_for_future(**kwargs):
    reconnects.append(kwargs["original_name"])
    return True

  async def forbidden_replay(**_kwargs):
    raise AssertionError("Sheets mutations must not use generic reconnect replay")

  manager._call_tool_once = fail_once
  manager._reconnect_stdio_server_for_future = reconnect_only_for_future
  manager._retry_stdio_tool_call_after_reconnect = forbidden_replay

  result, error = _run(manager.call_tool(
    "gsheets_write_range",
    {"spreadsheet": "sheet", "range": "Data!A1", "values": [[1]]},
  ))

  assert result is None
  assert error["sub_code"] == "mutation_outcome_uncertain"
  assert error["data"]["error"]["outcome"] == {
    "state": "uncertain",
    "phase": "dispatch",
    "mutation_may_have_occurred": True,
  }
  assert error["data"]["error"]["retry"]["safe"] is False
  assert error["data"]["error"]["retry"]["automatic"] is False
  assert "upstream detail" not in str(error)
  assert dispatches == ["gsheets_write_range"]
  assert reconnects == ["gsheets_write_range"]


def test_generic_stdio_sheets_mutation_timeout_reconnects_without_replay() -> None:
  manager = McpClientManager(config_path=None)
  server = _ServerState(
    name="gsheets-mcp",
    session=object(),
    exit_contexts=[object()],
    tool_definitions=[],
    tool_names={"gsheets_append_rows"},
    config={"type": "stdio", "command": "gsheets-mcp"},
  )
  manager._servers = {"gsheets-mcp": server}
  manager._tool_to_server = {"gsheets_append_rows": "gsheets-mcp"}
  dispatches = []
  reconnects = []

  async def time_out(**kwargs):
    dispatches.append(kwargs["original_name"])
    raise asyncio.TimeoutError("timeout with upstream detail")

  async def reconnect_only_for_future(**kwargs):
    reconnects.append(kwargs["original_name"])
    return True

  async def forbidden_replay(**_kwargs):
    raise AssertionError("Sheets mutations must not replay after a timeout")

  manager._call_tool_once = time_out
  manager._reconnect_stdio_server_for_future = reconnect_only_for_future
  manager._retry_stdio_tool_call_after_reconnect = forbidden_replay

  result, error = _run(manager.call_tool(
    "gsheets_append_rows",
    {"spreadsheet": "sheet", "range": "Data!A1", "values": [[1]]},
  ))

  assert result is None
  assert error["sub_code"] == "mutation_outcome_uncertain"
  assert error["data"]["error"]["outcome"] == {
    "state": "uncertain",
    "phase": "dispatch",
    "mutation_may_have_occurred": True,
  }
  assert error["data"]["error"]["retry"]["safe"] is False
  assert error["data"]["error"]["retry"]["automatic"] is False
  assert "upstream detail" not in str(error)
  assert dispatches == ["gsheets_append_rows"]
  assert reconnects == ["gsheets_append_rows"]


def test_sheets_requires_direct_structured_result_but_other_servers_keep_json_fallback() -> None:
  text_result = SimpleNamespace(
    isError=False,
    structuredContent=None,
    content=[SimpleNamespace(text='{"status":"ok","value":1}')],
  )

  sheets = McpClientManager(config_path=None)
  sheets._servers = {"gsheets-mcp": SimpleNamespace(session=object(), config={"type": "stdio"})}
  sheets._tool_to_server = {"gsheets_read_range": "gsheets-mcp"}
  sheets._call_tool_once = lambda **_: asyncio.sleep(0, result=text_result)

  sheets_result, sheets_error = _run(sheets.call_tool("gsheets_read_range", {}))

  assert sheets_result is None
  assert sheets_error["sub_code"] == "invalid_sheets_result_contract"
  assert sheets_error["data"]["error"]["outcome"]["state"] == "unchanged"

  other = McpClientManager(config_path=None)
  other._servers = {"other-mcp": SimpleNamespace(session=object(), config={"type": "stdio"})}
  other._tool_to_server = {"other_read": "other-mcp"}
  other._call_tool_once = lambda **_: asyncio.sleep(0, result=text_result)

  other_result, other_error = _run(other.call_tool("other_read", {}))

  assert other_error is None
  assert other_result == {"status": "ok", "value": 1}
