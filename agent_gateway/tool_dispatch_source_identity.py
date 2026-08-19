"""Gateway-local reader for the declarative ``source_identity`` descriptors.

The dispatch record settles at the tool boundary inside ``agent_gateway``,
which never statically imports ``agent.*`` (policy reads go through
:mod:`agent_gateway.policy_imports` soft imports).  The api-side extraction
chain in ``agent.shared.citation_source_extractors`` therefore cannot be
called from here, so :mod:`agent_gateway.tool_dispatch_declarations` declares
*what* a tool's source identity looks like and this module interprets that
declaration.

Only the identity triple survives the boundary — ``document_id``,
``source_kind`` and ``source_url`` — because that is exactly the compaction
the evidence fold consumes (the api side compacts to the same triple before
emitting a ``source_observation`` block).  Everything the api extractors also
compute (offsets, snippets, citation readiness, raw citation items) belongs to
the citation-minting path and is deliberately not duplicated here.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import re
from typing import Any


# --------------------------------------------------------------------------
# Canonical identity vocabulary (mirrors ``research.source_identity``; the
# gateway cannot import it, so the accepted shapes are restated as the
# boundary's own contract and pinned by the golden-parity test).
# --------------------------------------------------------------------------

SEC_ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")
CANONICAL_EDGAR_DOCUMENT_ID_RE = re.compile(r"^edgar:(\d{10}-\d{2}-\d{6})$")
CANONICAL_EDGAR_DOCUMENT_FILE_ID_RE = re.compile(
  r"^edgar:(\d{10}-\d{2}-\d{6})/[A-Za-z0-9._-]+\.[A-Za-z0-9]+$"
)
EDGAR_DOCUMENT_ID_WITH_LOCATOR_RE = re.compile(
  r"^edgar:(\d{10}-\d{2}-\d{6})(?::.+)$", re.IGNORECASE
)
LIVE_EDGAR_DOCUMENT_ID_RE = re.compile(
  r"^edgar:(?:[A-Z]{1,16}|\d{10}):(?:10-K|10-Q|8-K|DEF14A|DEF 14A|20-F|6-K):"
  r"(\d{10}-\d{2}-\d{6})(?::.+)?$",
  re.IGNORECASE,
)
CANONICAL_TRANSCRIPT_DOCUMENT_ID_RE = re.compile(
  r"^fmp_transcripts:[A-Z][A-Z0-9.-]{0,15}_[0-9]{4}-Q[1-4]$"
)
CANONICAL_DOC_DOCUMENT_ID_RE = re.compile(r"^doc:[0-9a-f]{32}$")
CANONICAL_WEB_DOCUMENT_ID_RE = re.compile(r"^web:([0-9a-fA-F]{64})$")

_CANONICAL_EDGAR_DOCUMENT_ACCESSION_RE = re.compile(
  r"^edgar:(\d{10}-\d{2}-\d{6})(?:/[A-Za-z0-9._-]+\.[A-Za-z0-9]+)?$"
)


def _text(value: Any) -> str | None:
  if value is None:
    return None
  normalized = str(value).strip()
  return normalized or None


def _required_str(value: Any) -> str | None:
  if value is None:
    return None
  text = str(value).strip()
  return text or None


def _optional_str(value: Any) -> str | None:
  if value is None:
    return None
  text = str(value)
  return text if text != "" else None


def _first_present(*values: Any) -> Any:
  for value in values:
    if value is None or value == "":
      continue
    return value
  return None


def _first_non_empty_field_text(payload: Mapping[str, Any], *keys: str) -> str | None:
  for key in keys:
    value = _optional_str(payload.get(key))
    if value:
      return value
  return None


def canonicalize_filing_document_id(value: Any) -> str | None:
  normalized = _text(value)
  if normalized is None:
    return None
  if SEC_ACCESSION_RE.fullmatch(normalized):
    return f"edgar:{normalized}"
  if CANONICAL_EDGAR_DOCUMENT_ID_RE.fullmatch(normalized) is not None:
    return normalized
  if CANONICAL_EDGAR_DOCUMENT_FILE_ID_RE.fullmatch(normalized) is not None:
    return normalized
  match = EDGAR_DOCUMENT_ID_WITH_LOCATOR_RE.fullmatch(normalized)
  if match is not None:
    return f"edgar:{match.group(1)}"
  match = LIVE_EDGAR_DOCUMENT_ID_RE.fullmatch(normalized)
  if match is not None:
    return f"edgar:{match.group(1)}"
  return None


def canonicalize_transcript_document_id(value: Any) -> str | None:
  normalized = _text(value)
  if normalized is None:
    return None
  return (
    normalized
    if CANONICAL_TRANSCRIPT_DOCUMENT_ID_RE.fullmatch(normalized)
    else None
  )


def canonicalize_doc_document_id(value: Any) -> str | None:
  normalized = _text(value)
  if normalized is None:
    return None
  return normalized if CANONICAL_DOC_DOCUMENT_ID_RE.fullmatch(normalized) else None


def canonicalize_web_document_id(value: Any) -> str | None:
  normalized = _text(value)
  if normalized is None:
    return None
  match = CANONICAL_WEB_DOCUMENT_ID_RE.fullmatch(normalized)
  if match is None:
    return None
  return f"web:{match.group(1).lower()}"


def normalize_citation_document_id(source_kind: Any, document_id: Any) -> str | None:
  normalized_kind = str(source_kind or "").strip().lower()
  normalized_id = _text(document_id)
  if normalized_id is None:
    return None
  if normalized_kind == "vendor":
    return normalized_id if normalized_id.lower().startswith("fmp:") else None
  if normalized_kind == "web":
    return canonicalize_web_document_id(normalized_id)
  if "transcript" in normalized_kind:
    return canonicalize_transcript_document_id(normalized_id)
  if normalized_kind in {"document", "doc"}:
    return canonicalize_doc_document_id(normalized_id)
  if normalized_kind in {
    "filing",
    "edgar",
    "filing_document",
    "filing_evidence",
    "section_text",
    "concept_citation",
  }:
    return canonicalize_filing_document_id(normalized_id)
  return (
    canonicalize_filing_document_id(normalized_id)
    or canonicalize_transcript_document_id(normalized_id)
    or canonicalize_doc_document_id(normalized_id)
    or canonicalize_web_document_id(normalized_id)
  )


# --------------------------------------------------------------------------
# Shared shape helpers
# --------------------------------------------------------------------------


def _as_mapping(value: Any) -> dict[str, Any]:
  return dict(value) if isinstance(value, Mapping) else {}


def _identity(
  *,
  document_id: Any,
  source_kind: Any,
  source_url: Any = None,
) -> dict[str, Any] | None:
  """Build one compacted identity record, rejecting malformed identities."""

  document_text = _required_str(document_id)
  kind_text = _required_str(source_kind)
  if not document_text or not kind_text:
    return None
  record: dict[str, Any] = {
    "document_id": document_text,
    "source_kind": kind_text,
  }
  url_text = _required_str(source_url)
  if url_text:
    record["source_url"] = url_text
  return record


def _source_url_from_payload(
  *payloads: Mapping[str, Any],
  prefer_url: bool = True,
) -> str | None:
  keys = (
    ("url", "source_url", "sourceUrl", "filing_url", "filingUrl")
    if prefer_url
    else ("source_url", "sourceUrl", "filing_url", "filingUrl", "url")
  )
  for payload in payloads:
    value = _first_non_empty_field_text(payload, *keys)
    if value:
      return value
  return None


def _normalize_search_hit_source_kind(
  value: str | None,
  *,
  default_source_kind: str,
  preserve_unknown: bool,
) -> str:
  default_kind = str(default_source_kind or "").strip() or "filing"
  raw = str(value or "").strip()
  if not raw:
    return default_kind
  normalized = raw.lower().replace("-", "_")
  default_normalized = default_kind.lower().replace("-", "_")
  if normalized == default_normalized:
    return default_kind
  if default_normalized == "filing" and normalized in {
    "edgar",
    "sec",
    "sec_edgar",
    "sec_filings",
    "filings",
  }:
    return "filing"
  if default_normalized == "transcript" and normalized in {
    "earnings_call",
    "earnings_transcript",
    "fmp_transcripts",
    "transcripts",
  }:
    return "transcript"
  return raw if preserve_unknown else default_kind


def _search_hit_source_kind(
  hit: Mapping[str, Any],
  *,
  default_source_kind: str,
) -> str:
  explicit = _optional_str(hit.get("source_kind"))
  if explicit:
    return _normalize_search_hit_source_kind(
      explicit,
      default_source_kind=default_source_kind,
      preserve_unknown=True,
    )
  return _normalize_search_hit_source_kind(
    _optional_str(hit.get("source")),
    default_source_kind=default_source_kind,
    preserve_unknown=False,
  )


# --------------------------------------------------------------------------
# Parser identity resolution
# --------------------------------------------------------------------------


def _filing_accession_component(document_id: str | None) -> str | None:
  canonical = canonicalize_filing_document_id(document_id)
  if canonical is None:
    return None
  match = _CANONICAL_EDGAR_DOCUMENT_ACCESSION_RE.fullmatch(canonical)
  return match.group(1) if match is not None else None


def _more_specific_filing_document_id(first: str, second: str) -> str:
  if "/" in first:
    return first
  if "/" in second:
    return second
  return first


def _parser_document_id(
  raw_document_id: str | None,
  accession: str | None,
) -> str | None:
  accession_document_id = canonicalize_filing_document_id(accession)
  raw_canonical_id = canonicalize_filing_document_id(raw_document_id)
  if accession_document_id is not None and raw_canonical_id is not None:
    accession_component = _filing_accession_component(accession_document_id)
    raw_component = _filing_accession_component(raw_canonical_id)
    if (
      accession_component is None
      or raw_component is None
      or accession_component != raw_component
    ):
      return None
    return _more_specific_filing_document_id(raw_canonical_id, accession_document_id)
  return accession_document_id or raw_canonical_id


def _consistent_parser_document_id(*payloads: Mapping[str, Any]) -> str | None:
  resolved: str | None = None
  saw_identity = False
  for payload in payloads:
    raw_document_id = _optional_str(payload.get("document_id"))
    accession = _optional_str(payload.get("accession"))
    if raw_document_id is None and accession is None:
      continue
    saw_identity = True
    candidate = _parser_document_id(raw_document_id, accession)
    if candidate is None:
      return None
    if resolved is None:
      resolved = candidate
      continue
    resolved = _parser_document_id(candidate, resolved)
    if resolved is None:
      return None
  return resolved if saw_identity else None


def _source_id_token(value: Any) -> str:
  text = str(value or "").strip()
  token = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
  return token or "unknown"


def _period_from_query(query: Mapping[str, Any]) -> str | None:
  year = query.get("year")
  quarter = query.get("quarter")
  if year is None or quarter is None:
    return None
  return f"{year}-Q{quarter}"


def _parser_fallback_document_id(
  item: Mapping[str, Any],
  filing: Mapping[str, Any],
  query: Mapping[str, Any],
) -> str | None:
  ticker = _optional_str(
    _first_present(item.get("ticker"), filing.get("ticker"), query.get("ticker"))
  )
  period = _optional_str(
    _first_present(
      item.get("fiscal_period"),
      item.get("period"),
      _period_from_query(query),
    )
  )
  form_type = _optional_str(
    _first_present(
      item.get("form_type"),
      item.get("filing_type"),
      filing.get("form_type"),
      filing.get("form"),
      query.get("form_type"),
    )
  )
  if not ticker or not (period or form_type):
    return None
  parts = [
    "parser-filing",
    _source_id_token(ticker),
    _source_id_token(form_type or "unknown-form"),
  ]
  if period:
    parts.append(_source_id_token(period))
  return ":".join(parts)


def _first_source_citation(item: Mapping[str, Any]) -> dict[str, Any]:
  citations = item.get("source_citations")
  if not isinstance(citations, list):
    return {}
  for citation in citations:
    if isinstance(citation, Mapping):
      return dict(citation)
  return {}


def _parser_item_identity(
  item: Mapping[str, Any],
  *,
  default_source_kind: str,
  filing: Mapping[str, Any] | None = None,
  query: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
  filing = filing or {}
  query = query or {}
  nested_citation = _first_source_citation(item)

  accession = _optional_str(
    _first_present(
      item.get("accession"),
      nested_citation.get("accession"),
      filing.get("accession"),
      query.get("accession"),
    )
  )
  raw_document_id = _optional_str(
    _first_present(item.get("document_id"), nested_citation.get("document_id"))
  )
  document_id = _parser_document_id(raw_document_id, accession)
  if document_id is None:
    if raw_document_id is not None or accession is not None:
      return None
    document_id = _parser_fallback_document_id(item, filing, query)
  if document_id is None:
    return None

  source_url = _optional_str(
    _first_present(
      item.get("source_url"),
      item.get("document_url"),
      item.get("filing_url"),
      item.get("url"),
      nested_citation.get("source_url"),
      nested_citation.get("document_url"),
      nested_citation.get("filing_url"),
      nested_citation.get("url"),
      filing.get("filing_url"),
      filing.get("url"),
    )
  )
  return _identity(
    document_id=document_id,
    source_kind=_optional_str(item.get("source_kind")) or default_source_kind,
    source_url=source_url,
  )


# --------------------------------------------------------------------------
# Metric identity resolution
# --------------------------------------------------------------------------


def _metric_citation_items(
  match: Mapping[str, Any],
) -> list[tuple[str | None, Mapping[str, Any]]]:
  citations = match.get("citations")
  if isinstance(citations, Mapping):
    return [
      (_optional_str(key), citation)
      for key, citation in citations.items()
      if isinstance(citation, Mapping)
    ]
  if isinstance(citations, list):
    return [(None, citation) for citation in citations if isinstance(citation, Mapping)]
  citation = match.get("citation")
  return [(None, citation)] if isinstance(citation, Mapping) else []


def _metric_document_id(
  citation: Mapping[str, Any],
  *,
  source_kind: str,
) -> str | None:
  document_ids: list[str] = []
  for key in ("document_id", "source_id", "accession"):
    raw = _optional_str(citation.get(key))
    if raw is None:
      continue
    normalized = normalize_citation_document_id(source_kind, raw)
    if normalized is None:
      return None
    document_ids.append(normalized)
  if not document_ids:
    return None
  first = document_ids[0]
  if any(document_id != first for document_id in document_ids[1:]):
    return None
  return first


# --------------------------------------------------------------------------
# Descriptor readers
# --------------------------------------------------------------------------


def _read_web_fetch(result: Mapping[str, Any]) -> list[dict[str, Any]]:
  final_url = _required_str(result.get("final_url"))
  content_hash = _required_str(result.get("content_hash"))
  text = _optional_str(result.get("text"))
  if not final_url or not content_hash or not text:
    return []
  digest = hashlib.sha256(final_url.encode("utf-8")).hexdigest()
  record = _identity(
    document_id=f"web:{digest}",
    source_kind="web",
    source_url=final_url,
  )
  return [record] if record is not None else []


def _read_search_hits(
  result: Mapping[str, Any],
  descriptor: Mapping[str, Any],
) -> list[dict[str, Any]]:
  container = str(descriptor.get("container") or "hits")
  default_source_kind = str(descriptor.get("default_source_kind") or "filing")
  value = result.get(container)
  if not isinstance(value, list) or not value:
    return []
  identities: list[dict[str, Any]] = []
  for hit in value:
    if not isinstance(hit, Mapping):
      continue
    document_id = _required_str(hit.get("document_id"))
    if document_id is None:
      continue
    document_id = normalize_citation_document_id(default_source_kind, document_id)
    if document_id is None:
      continue
    record = _identity(
      document_id=document_id,
      source_kind=_search_hit_source_kind(
        hit,
        default_source_kind=default_source_kind,
      ),
      source_url=_optional_str(hit.get("source_url")),
    )
    if record is not None:
      identities.append(record)
  return identities


def _read_documents(
  result: Mapping[str, Any],
  descriptor: Mapping[str, Any],
) -> list[dict[str, Any]]:
  container = str(descriptor.get("container") or "documents")
  source_kind = str(descriptor.get("source_kind") or "filing")
  value = result.get(container)
  if not isinstance(value, list) or not value:
    return []
  identities: list[dict[str, Any]] = []
  for document in value:
    if not isinstance(document, Mapping):
      continue
    document_id = _required_str(document.get("document_id"))
    if document_id is None:
      continue
    document_id = normalize_citation_document_id(source_kind, document_id)
    if document_id is None:
      continue
    record = _identity(
      document_id=document_id,
      source_kind=source_kind,
      source_url=_source_url_from_payload(document, prefer_url=False),
    )
    if record is not None:
      identities.append(record)
  return identities


def _read_single_document(
  result: Mapping[str, Any],
  descriptor: Mapping[str, Any],
) -> list[dict[str, Any]]:
  source_kind = str(descriptor.get("source_kind") or "filing")
  document_id = _required_str(result.get("document_id"))
  if document_id is None:
    return []
  document_id = normalize_citation_document_id(source_kind, document_id)
  if document_id is None:
    return []
  record = _identity(
    document_id=document_id,
    source_kind=source_kind,
    source_url=_source_url_from_payload(result),
  )
  return [record] if record is not None else []


def _read_parser_items(
  result: Mapping[str, Any],
  descriptor: Mapping[str, Any],
) -> list[dict[str, Any]]:
  container = str(descriptor.get("container") or "hits")
  default_source_kind = str(descriptor.get("default_source_kind") or "filing")
  value = result.get(container)
  if not isinstance(value, list) or not value:
    return []
  identities: list[dict[str, Any]] = []
  for item in value:
    if not isinstance(item, Mapping):
      continue
    record = _parser_item_identity(item, default_source_kind=default_source_kind)
    if record is not None:
      identities.append(record)
  return identities


def _read_parser_document(result: Mapping[str, Any]) -> list[dict[str, Any]]:
  filing = _as_mapping(result.get("filing"))
  query = _as_mapping(result.get("query"))
  record = _parser_item_identity(
    result,
    default_source_kind="filing_document",
    filing=filing,
    query=query,
  )
  return [record] if record is not None else []


def _read_parser_filings(result: Mapping[str, Any]) -> list[dict[str, Any]]:
  filings = result.get("filings")
  if not isinstance(filings, list) or not filings:
    return []
  query = {
    "ticker": result.get("ticker"),
    "year": result.get("year"),
    "quarter": result.get("quarter"),
  }
  identities: list[dict[str, Any]] = []
  for filing in filings:
    if not isinstance(filing, Mapping):
      continue
    document_id = _consistent_parser_document_id(filing)
    if document_id is None:
      continue
    item = {**dict(filing), "document_id": document_id, "source_kind": "filing"}
    record = _parser_item_identity(
      item,
      default_source_kind="filing",
      filing=filing,
      query=query,
    )
    if record is not None:
      identities.append(record)
  return identities


def _read_parser_filing_sections(result: Mapping[str, Any]) -> list[dict[str, Any]]:
  sections = result.get("sections")
  if not isinstance(sections, Mapping) or not sections:
    return []
  filing = _as_mapping(result.get("filing"))
  query = {
    "ticker": result.get("ticker"),
    "year": result.get("year"),
    "quarter": result.get("quarter"),
    "form_type": _first_present(result.get("filing_type"), result.get("form")),
  }
  identities: list[dict[str, Any]] = []
  for raw_section_key, section_payload in sections.items():
    section_key = _required_str(raw_section_key)
    if section_key is None or not isinstance(section_payload, Mapping):
      continue
    document_id = _consistent_parser_document_id(result, filing, section_payload)
    if document_id is None:
      continue
    raw_section_text = section_payload.get("text")
    section_text = raw_section_text if isinstance(raw_section_text, str) else None
    if section_text is not None and not section_text.strip():
      section_text = None
    if section_text is not None:
      content_sha256 = hashlib.sha256(section_text.encode("utf-8")).hexdigest()
      declared_content_sha256 = _optional_str(section_payload.get("content_sha256"))
      if (
        declared_content_sha256 is not None
        and declared_content_sha256 != content_sha256
      ):
        continue
    item = {
      **dict(section_payload),
      "document_id": document_id,
      "source_kind": "section_text",
    }
    record = _parser_item_identity(
      item,
      default_source_kind="section_text",
      filing=filing,
      query=query,
    )
    if record is not None:
      identities.append(record)
  return identities


def _read_metric_citations(result: Mapping[str, Any]) -> list[dict[str, Any]]:
  matches = result.get("matches")
  if not isinstance(matches, list) or not matches:
    return []
  identities: list[dict[str, Any]] = []
  for match in matches:
    if not isinstance(match, Mapping):
      continue
    for _citation_key, citation in _metric_citation_items(match):
      source_kind = _optional_str(citation.get("source_kind")) or "xbrl_fact"
      document_id = _metric_document_id(citation, source_kind=source_kind)
      if document_id is None:
        continue
      record = _identity(
        document_id=document_id,
        source_kind=source_kind,
        source_url=_source_url_from_payload(citation),
      )
      if record is not None:
        identities.append(record)
  return identities


_READERS = {
  "web_fetch": lambda result, descriptor: _read_web_fetch(result),
  "search_hits": _read_search_hits,
  "documents": _read_documents,
  "single_document": _read_single_document,
  "parser_items": _read_parser_items,
  "parser_document": lambda result, descriptor: _read_parser_document(result),
  "parser_filings": lambda result, descriptor: _read_parser_filings(result),
  "parser_filing_sections": (
    lambda result, descriptor: _read_parser_filing_sections(result)
  ),
  "metric_citations": lambda result, descriptor: _read_metric_citations(result),
}


class UnknownSourceIdentityDescriptor(ValueError):
  """Raised when a declaration names a descriptor kind with no reader."""


def read_source_identities(
  descriptor: Mapping[str, Any] | None,
  result: Any,
) -> tuple[dict[str, Any], ...]:
  """Interpret one declarative ``source_identity`` descriptor.

  Returns the compacted identity triples in the same order the api-side
  extraction chain produces them.  A tool with no declared descriptor
  contributes no identities — silence, never a guess.
  """

  if descriptor is None:
    return ()
  if not isinstance(result, Mapping):
    return ()
  kind = str(descriptor.get("kind") or "")
  reader = _READERS.get(kind)
  if reader is None:
    raise UnknownSourceIdentityDescriptor(
      f"unknown source_identity descriptor kind: {kind!r}"
    )
  return tuple(reader(result, descriptor))


__all__ = [
  "UnknownSourceIdentityDescriptor",
  "canonicalize_filing_document_id",
  "canonicalize_transcript_document_id",
  "canonicalize_web_document_id",
  "normalize_citation_document_id",
  "read_source_identities",
]
