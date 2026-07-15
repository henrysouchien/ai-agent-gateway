from __future__ import annotations

import copy
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
  "gsheets_list_tabs": (
    "Workflow: pass a Google Sheets URL or ID, then reuse the returned normalized spreadsheet and exact "
    "tab titles in later range operations."
  ),
  "gsheets_read_range": (
    "Workflow: list tabs when the layout is unknown, then read an A1 range. Reuse the returned spreadsheet "
    "and range unchanged in a later write, clear, or recalculation call."
  ),
  "gsheets_write_range": (
    "Workflow: inspect the target first when existing values matter, then provide a non-empty two-dimensional "
    "values array. A lost response can leave the write outcome uncertain, so do not blindly replay it."
  ),
  "gsheets_append_rows": (
    "Workflow: inspect the destination columns first, then append a non-empty two-dimensional values array. "
    "Append is non-idempotent; never repeat it solely because a response was lost."
  ),
  "gsheets_create_spreadsheet": (
    "Workflow: create once and reuse the returned spreadsheet ID. Creation is non-idempotent; a lost response "
    "must be investigated before another create call."
  ),
  "gsheets_copy_spreadsheet": (
    "Workflow: list source tabs before selecting an optional tabs subset. On a partial failure, inspect the "
    "returned destination recovery details instead of creating another copy."
  ),
  "gsheets_search_spreadsheets": (
    "Workflow: local mode only. Search by title, select one result, then pass its spreadsheet ID to the common "
    "Sheets tools."
  ),
  "gsheets_clear_range": (
    "Workflow: read the target range before clearing when its contents matter. The returned range identifies "
    "the cells confirmed cleared."
  ),
  "gsheets_recalculate_range": (
    "Workflow: use only for a range containing formulas and blanks. The tool verifies formula restoration and "
    "reports any compensation or recovery state explicitly."
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


@dataclass
class LogicalToolAliasResult:
  tool_definitions: list[dict[str, Any]]
  tool_to_server: dict[str, str]
  dispatch_to_original: dict[str, str]
  mcp_tool_names: set[str]
  logical_tool_definitions: dict[str, list[dict[str, Any]]]


def add_logical_tool_aliases(
  *,
  tool_definitions: list[dict[str, Any]],
  tool_to_server: dict[str, str],
  prefixed_to_original: dict[str, str],
  mcp_tool_names: set[str],
  logical_server_routes: Mapping[str, str],
  logical_tool_aliases: Mapping[str, Mapping[str, str]],
  policy_server_for_tool: Any | None,
  transport_only_servers: set[str] | None = None,
) -> LogicalToolAliasResult:
  merged = list(tool_definitions)
  owners = dict(tool_to_server)
  dispatch_names = dict(prefixed_to_original)
  names = set(mcp_tool_names)
  logical_definitions: dict[str, list[dict[str, Any]]] = {}
  transport_only = set(transport_only_servers or set())

  definitions_by_owner_and_original: dict[tuple[str, str], dict[str, Any]] = {}
  for tool_definition in tool_definitions:
    exposed_name = str(tool_definition.get("name") or "")
    owner = tool_to_server.get(exposed_name)
    if not owner:
      continue
    original_name = prefixed_to_original.get(exposed_name, exposed_name)
    definitions_by_owner_and_original[(owner, original_name)] = tool_definition

  logical_catalog_servers = list(logical_tool_aliases)
  logical_catalog_servers.extend(
    logical_server
    for logical_server, physical_server in logical_server_routes.items()
    if physical_server in transport_only and logical_server not in logical_tool_aliases
  )
  for logical_server in logical_catalog_servers:
    aliases = logical_tool_aliases.get(logical_server, {})
    physical_server = logical_server_routes.get(logical_server)
    if not physical_server:
      raise ValueError(f"logical tool aliases require a physical route for {logical_server}")
    physical_definitions = {
      original_name
      for owner, original_name in definitions_by_owner_and_original
      if owner == physical_server
    }
    if not physical_definitions:
      # A logical catalog is available only when its physical transport
      # connected. Preserve partial gateway startup when that adapter is down.
      continue

    if physical_server in transport_only:
      aliases_by_original: dict[str, str] = {}
      for alias_name, original_name in aliases.items():
        existing_alias = aliases_by_original.get(original_name)
        if existing_alias is not None and existing_alias != alias_name:
          raise ValueError(
            f"physical tool {physical_server}.{original_name} has multiple logical aliases: "
            f"{existing_alias}, {alias_name}"
          )
        source = definitions_by_owner_and_original.get((physical_server, original_name))
        if source is None:
          raise ValueError(
            f"logical tool alias {logical_server}.{alias_name} references unavailable "
            f"physical tool {physical_server}.{original_name}"
          )
        if policy_server_for_tool is not None:
          policy_owner = policy_server_for_tool(alias_name)
          if policy_owner != logical_server:
            raise ValueError(
              f"logical tool alias {alias_name} policy owner is {policy_owner!r}; "
              f"expected {logical_server!r}"
            )
        aliases_by_original[original_name] = alias_name

      physical_exposed_names = {
        exposed_name
        for exposed_name, owner in tool_to_server.items()
        if owner == physical_server
      }
      available_names = names - physical_exposed_names
      promoted_definitions: list[dict[str, Any]] = []
      promoted_by_physical_name: dict[str, dict[str, Any]] = {}
      promoted_owners: dict[str, str] = {}
      promoted_dispatch_names: dict[str, str] = {}

      for tool_definition in tool_definitions:
        physical_exposed_name = str(tool_definition.get("name") or "")
        if tool_to_server.get(physical_exposed_name) != physical_server:
          continue
        original_name = prefixed_to_original.get(
          physical_exposed_name,
          physical_exposed_name,
        )
        alias_name = aliases_by_original.get(original_name, original_name)
        if alias_name in available_names:
          raise ValueError(f"logical tool alias collides with an exposed tool: {alias_name}")
        if policy_server_for_tool is not None and original_name not in aliases_by_original:
          policy_owner = policy_server_for_tool(alias_name)
          if policy_owner not in {None, logical_server}:
            raise ValueError(
              f"logical identity tool {alias_name} policy owner is {policy_owner!r}; "
              f"expected {logical_server!r}"
            )

        alias_definition = copy.deepcopy(tool_definition)
        alias_definition["name"] = alias_name
        if alias_name != original_name:
          original_description = str(alias_definition.get("description") or "").strip()
          alias_note = f"Provider-neutral tool routed through the {logical_server} logical server."
          alias_definition["description"] = (
            f"{original_description}\n\n{alias_note}" if original_description else alias_note
          )
        promoted_definitions.append(alias_definition)
        promoted_by_physical_name[physical_exposed_name] = alias_definition
        promoted_owners[alias_name] = logical_server
        promoted_dispatch_names[alias_name] = original_name
        available_names.add(alias_name)

      merged = [
        promoted_by_physical_name.get(
          str(tool_definition.get("name") or ""),
          tool_definition,
        )
        for tool_definition in merged
        if (
          tool_to_server.get(str(tool_definition.get("name") or "")) != physical_server
          or str(tool_definition.get("name") or "") in promoted_by_physical_name
        )
      ]
      for exposed_name in physical_exposed_names:
        owners.pop(exposed_name, None)
        dispatch_names.pop(exposed_name, None)
      owners.update(promoted_owners)
      dispatch_names.update(promoted_dispatch_names)
      names = available_names
      logical_definitions[logical_server] = promoted_definitions
      continue

    for alias_name, original_name in aliases.items():
      if alias_name in names:
        raise ValueError(f"logical tool alias collides with an exposed tool: {alias_name}")
      source = definitions_by_owner_and_original.get((physical_server, original_name))
      if source is None:
        raise ValueError(
          f"logical tool alias {logical_server}.{alias_name} references unavailable "
          f"physical tool {physical_server}.{original_name}"
        )
      if policy_server_for_tool is not None:
        policy_owner = policy_server_for_tool(alias_name)
        if policy_owner != logical_server:
          raise ValueError(
            f"logical tool alias {alias_name} policy owner is {policy_owner!r}; "
            f"expected {logical_server!r}"
          )

      alias_definition = copy.deepcopy(source)
      alias_definition["name"] = alias_name
      original_description = str(alias_definition.get("description") or "").strip()
      alias_note = (
        f"Provider-neutral alias for {original_name}; routed through the "
        f"{logical_server} logical server."
      )
      alias_definition["description"] = (
        f"{original_description}\n\n{alias_note}" if original_description else alias_note
      )
      merged.append(alias_definition)
      owners[alias_name] = logical_server
      dispatch_names[alias_name] = original_name
      names.add(alias_name)
      logical_definitions.setdefault(logical_server, []).append(alias_definition)

  return LogicalToolAliasResult(
    tool_definitions=merged,
    tool_to_server=owners,
    dispatch_to_original=dispatch_names,
    mcp_tool_names=names,
    logical_tool_definitions=logical_definitions,
  )


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


__all__ = [
  "CollisionFilterResult",
  "LogicalToolAliasResult",
  "add_logical_tool_aliases",
  "apply_collision_filtering",
]
