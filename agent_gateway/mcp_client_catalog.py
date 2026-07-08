from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping


_MCP_TOOL_DESCRIPTION_GUIDANCE: dict[str, str] = {
  "filings_read": (
    "Argument guidance: read from corpus search/list results only. Copy document_id and section "
    "from filings_search or filings_list results. For text windows, copy char_start/char_end from "
    "filings_search hits and set offset_frame='document'; omit offsets if uncertain. Do not invent "
    "or normalize section names such as 'Notes to Consolidated Financial Statements' or "
    "'Part I, Item 8'; use the exact returned section, or omit section/offsets and rediscover with "
    "filings_search/filings_list. Do not pass transcript document_ids such as fmp_transcripts:*; "
    "use transcripts_read for transcript results."
  ),
  "transcripts_read": (
    "Argument guidance: read from transcript search/list results only. Copy document_id, section, "
    "char_start, char_end, and offset_frame from transcripts_search or transcripts_list results. "
    "The only supported section values are prepared_remarks and qa; omit section if uncertain. "
    "Do not use filings_read for transcript document_ids. With search-hit offsets, preserve the "
    "returned offset_frame; use offset_frame='scoped' only for offsets relative to the selected "
    "transcript section/window."
  ),
  "get_filings": (
    "Argument guidance: this is not a broad filing-list endpoint. Call only after deriving a "
    "fiscal/reporting integer year and integer quarter, then pass ticker, year, quarter, and "
    "optional source when selecting supported modes such as 8k, 20f, or 6k. Do not pass form_type, "
    "form_types, limit, period, fiscal_year, or quarter='FY'. For discovery/listing, use corpus "
    "filings_search/filings_list or a parser discovery tool before get_filings."
  ),
  "get_filing_tables": (
    "Argument guidance: requires ticker plus integer year and quarter. Optional filters are "
    "singular section, table_id, source, and accession. Do not pass form_type, limit, query, "
    "text, period, fiscal_period, sections, or table_type; sections is invalid, use section."
  ),
  "search_filing_tables": (
    "Argument guidance: use description for the table text/keyword search, period_from/period_to "
    "for date bounds, and section_key for section filters. Do not pass query, text, year, quarter, "
    "form_type, fiscal_period, or period. After metadata search, call get_filing_tables with the "
    "returned table_id to read table contents."
  ),
  "get_filing_document": (
    "Argument guidance: read filing text from parser filing coordinates only. Do not pass corpus "
    "document_id values, raw edgar: accessions, table_id/table_ids, offset_frame, output, or excerpt "
    "window arguments. For specific sections, first call get_filing_sections and copy the exact "
    "returned section key/name; do not invent topical section names such as outlook, director "
    "nominees, use_of_proceeds, or normalized Item labels. For topical text search, use "
    "search_filing_text. For tables, use search_filing_tables/get_filing_tables. For cited excerpt "
    "windows, use filings_read with a document_id and offsets copied from corpus results."
  ),
  "get_event_filings": (
    "Argument guidance: use for event-driven filings only, such as 8-K, proxy, offering, merger, "
    "or ownership-related filings. Do not request periodic 10-K or 10-Q filings here; for periodic "
    "filings use get_filings or corpus filings_search/filings_list, then read content with "
    "get_filing_document/get_filing_sections or filings_read."
  ),
  "get_model_insights": (
    "Argument guidance: use only for internally modeled ModelInsights readbacks. Pass research_file_id, "
    "optional model_insights_id, and optional format='agent'. Do not pass ticker or query; if you only "
    "have a ticker, first resolve the typed research file through research-corpus-mcp.thesis_list or "
    "research-corpus-mcp.list_research_files, then retry with the returned research_file_id."
  ),
  "industry_peer_comparison": (
    "Argument guidance: Pass symbol such as symbol='MSCI'. Do not pass ticker. Use this for portfolio "
    "peer/industry comparison readbacks. For custom peer sets or metric-specific comparisons, use "
    "compare_peers with its schema-supported symbol and peer arguments."
  ),
  "get_price_target": (
    "Argument guidance: use only for internally modeled PriceTarget readbacks. Pass research_file_id, "
    "optional handoff_id, and optional format='agent'. Do not pass ticker; if you only have a ticker, "
    "first resolve the typed research file through research-corpus-mcp.thesis_list or "
    "research-corpus-mcp.list_research_files, then retry with the returned research_file_id. Do not use "
    "this tool for sell-side analyst consensus or external price-target spreads."
  ),
}


_MCP_TOOL_PROPERTY_DESCRIPTION_GUIDANCE: dict[str, dict[str, str]] = {
  "filings_read": {
    "document_id": (
      "Exact document_id copied from filings_search or filings_list. Do not synthesize it. "
      "Transcript ids such as fmp_transcripts:* belong in transcripts_read."
    ),
    "section": (
      "Exact section value returned by filings_search or filings_list. Do not invent normalized names "
      "such as Notes to Consolidated Financial Statements, Part I Item 8, or Item 2."
    ),
    "char_start": (
      "Optional integer offset copied from a search/list hit. With offset_frame='document', it must be "
      "a document offset inside the selected scope; omit if uncertain."
    ),
    "char_end": (
      "Optional integer offset copied from a search/list hit. Use with the same offset_frame as the hit; "
      "omit if uncertain."
    ),
    "offset_frame": (
      "Use 'document' for offsets from search hits. Use 'scoped' only for offsets known to be relative "
      "to the selected section/window."
    ),
  },
  "transcripts_read": {
    "document_id": (
      "Exact transcript document_id copied from transcripts_search or transcripts_list, usually "
      "prefixed fmp_transcripts:. Do not pass this to filings_read."
    ),
    "section": "Optional transcript section enum: prepared_remarks or qa only. Omit if uncertain.",
    "char_start": (
      "Optional integer offset copied from a transcript hit. Use with the same offset_frame as the hit; "
      "omit if uncertain."
    ),
    "char_end": (
      "Optional integer offset copied from a transcript hit. Use with the same offset_frame as the hit; "
      "omit if uncertain."
    ),
    "offset_frame": (
      "Use the offset_frame returned by transcripts_search/list. Use 'scoped' only for section/window-"
      "relative offsets; do not mix document offsets with scoped offsets."
    ),
  },
  "get_filings": {
    "ticker": "Ticker or SEC symbol only. Do not use this tool for broad discovery without year/quarter.",
    "year": "Required integer fiscal/reporting year, for example 2024. Do not omit or pass strings like FY2024.",
    "quarter": "Required integer quarter from 1 to 4. For annual filings, use the parser's required integer quarter, not 'FY'.",
    "source": "Optional supported filing source/mode such as 8k, 20f, or 6k. Do not use form_type.",
  },
  "get_filing_tables": {
    "year": "Integer fiscal year, for example 2024. Do not pass strings like '2024-FY'.",
    "quarter": "Integer fiscal quarter from 1 to 4. Do not pass string values.",
    "section": "Optional singular section filter. Do not pass sections.",
    "table_id": "Optional table_id returned by search_filing_tables to read one table.",
  },
  "search_filing_tables": {
    "description": "Plain text table description or keyword query. Use this instead of query or text.",
    "period_from": "Optional lower period/date bound. Do not use fiscal_period or period.",
    "period_to": "Optional upper period/date bound. Do not use fiscal_period or period.",
    "section_key": "Optional section filter key. Do not use section, sections, or form_type.",
  },
  "get_filing_document": {
    "section": (
      "Exact section key/name returned by get_filing_sections. Do not pass topical labels or normalized "
      "aliases; use search_filing_text for topical search."
    ),
    "sections": (
      "Exact section keys/names returned by get_filing_sections. Do not pass topical labels or normalized "
      "aliases; use search_filing_text for topical search."
    ),
    "ticker": "Ticker or SEC symbol plus parser period/form coordinates, not a corpus document_id.",
    "accession": "Use only a parser-supported accession value. Do not pass raw edgar: document_id strings.",
  },
  "get_event_filings": {
    "form_type": "Event filing form only, for example 8-K or proxy forms. Do not pass 10-K or 10-Q.",
    "ticker": "Ticker or SEC symbol for event-filing discovery. Use get_filings for periodic 10-K/10-Q filings.",
  },
  "get_model_insights": {
    "research_file_id": (
      "Required typed research file id. This is not a ticker. Resolve it from thesis_list, list_research_files, "
      "or task context before calling get_model_insights."
    ),
    "model_insights_id": "Optional specific ModelInsights id; omit to read the latest current-model insights.",
    "format": "Optional output format such as 'agent'. This does not change the required research_file_id identity.",
  },
  "industry_peer_comparison": {
    "symbol": "Ticker/company symbol accepted by this tool. Do not pass ticker; use symbol.",
  },
  "get_price_target": {
    "research_file_id": (
      "Required typed research file id. This is not a ticker. Resolve it from thesis_list, list_research_files, "
      "or task context before calling get_price_target."
    ),
    "handoff_id": "Optional specific handoff id for the same research_file_id; omit for the current model handoff.",
    "format": "Optional output format such as 'agent'. This does not change the required research_file_id identity.",
  },
}


def tool_argument_guidance(tool_name: str) -> str | None:
  return _MCP_TOOL_DESCRIPTION_GUIDANCE.get(str(tool_name or "").strip())


@dataclass
class CollisionFilterResult:
  tool_definitions: list[dict[str, Any]]
  tool_to_server: dict[str, str]
  prefixed_to_original: dict[str, str]
  mcp_tool_names: set[str]


def apply_collision_filtering(
  *,
  servers: Mapping[str, Any],
  builtin_tool_names: set[str],
  strip_input_fields: set[str],
  logger: logging.Logger,
) -> CollisionFilterResult:
  existing_names = set(builtin_tool_names)
  seen_mcp_names: dict[str, str] = {}
  merged: list[dict[str, Any]] = []
  tool_to_server: dict[str, str] = {}
  prefixed_to_original: dict[str, str] = {}

  for server_name, state in servers.items():
    prefix = state.tool_prefix
    filtered: list[dict[str, Any]] = []
    filtered_names: set[str] = set()

    for tool in state.tool_definitions:
      original_name = tool["name"]
      tool_name = f"{prefix}{original_name}" if prefix else original_name
      if tool_name in existing_names:
        logger.warning(
          "Skipping MCP tool %s from %s: collides with built-in tool. "
          "Fix: add \"tool_prefix\" to the \"%s\" server config to namespace its tools.",
          tool_name,
          server_name,
          server_name,
        )
        continue

      first_server = seen_mcp_names.get(tool_name)
      if first_server:
        logger.warning(
          "Skipping MCP tool %s from %s: collides with MCP tool from %s. "
          "Fix: add \"tool_prefix\" to the \"%s\" server config to namespace its tools.",
          tool_name,
          server_name,
          first_server,
          server_name,
        )
        continue

      seen_mcp_names[tool_name] = server_name
      tool_to_server[tool_name] = server_name
      if prefix:
        prefixed_to_original[tool_name] = original_name
        tool = {**tool, "name": tool_name}
      _apply_tool_definition_guidance(tool, original_name)
      filtered.append(tool)
      filtered_names.add(tool_name)

    state.tool_definitions = filtered
    state.tool_names = filtered_names
    merged.extend(filtered)

    logger.info(
      "MCP server %s connected | %d tools: %s",
      server_name,
      len(filtered_names),
      sorted(filtered_names),
    )

  _strip_hidden_input_fields(merged, strip_input_fields)
  return CollisionFilterResult(
    tool_definitions=merged,
    tool_to_server=tool_to_server,
    prefixed_to_original=prefixed_to_original,
    mcp_tool_names=set(tool_to_server.keys()),
  )


def _strip_hidden_input_fields(
  tool_definitions: list[dict[str, Any]],
  strip_input_fields: set[str],
) -> None:
  if not strip_input_fields:
    return
  for tool_def in tool_definitions:
    schema = tool_def.get("input_schema", {})
    props = schema.get("properties", {})
    required = schema.get("required", [])
    for field in strip_input_fields:
      props.pop(field, None)
      if field in required:
        required.remove(field)


def _append_sentence(existing: Any, addition: str) -> str:
  base = str(existing or "").strip()
  if not base:
    return addition
  if addition in base:
    return base
  return f"{base}\n\n{addition}"


def _apply_tool_definition_guidance(tool: dict[str, Any], original_name: str) -> None:
  guidance = _MCP_TOOL_DESCRIPTION_GUIDANCE.get(original_name)
  if guidance:
    tool["description"] = _append_sentence(tool.get("description"), guidance)

  property_guidance = _MCP_TOOL_PROPERTY_DESCRIPTION_GUIDANCE.get(original_name)
  schema = tool.get("input_schema")
  if not property_guidance or not isinstance(schema, dict):
    return
  props = schema.get("properties")
  if not isinstance(props, dict):
    return
  for prop_name, prop_guidance in property_guidance.items():
    prop = props.get(prop_name)
    if isinstance(prop, dict):
      prop["description"] = _append_sentence(prop.get("description"), prop_guidance)


__all__ = ["CollisionFilterResult", "apply_collision_filtering"]
