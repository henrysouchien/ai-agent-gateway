"""Dependency-light canonical persisted-ticker identity contract."""

from __future__ import annotations

import re

from .models import ContractRef, sha256_digest


EXTENDED_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.]{0,14}$")
ENTITY_TICKER_RE = re.compile(r"^[A-Z0-9]{1,6}$")
TICKER_NORMALIZATION_ALGORITHM_ID = "canonical-contract-ticker-normalization/v1"

# Keep this tuple ordered: its canonical mapping is part of the typed-input
# contract digest shared by ordinary delegation and workflow admission.
TICKER_CONTRACT_SEMANTIC_MANIFEST = (
  ("namespace", "workflow"),
  ("name", "ticker"),
  ("version", "1.0"),
  ("canonical_encoding", "utf-8-json-string"),
  ("normalizer", TICKER_NORMALIZATION_ALGORITHM_ID),
  ("persisted_rule", "canonical-contract-ticker/v1"),
)

_TICKER_MANIFEST = dict(TICKER_CONTRACT_SEMANTIC_MANIFEST)
TICKER_INPUT_CONTRACT = ContractRef(
  namespace=_TICKER_MANIFEST["namespace"],
  name=_TICKER_MANIFEST["name"],
  version=_TICKER_MANIFEST["version"],
  digest=sha256_digest(_TICKER_MANIFEST),
)

# Longest suffixes first to keep boundary normalization deterministic (for
# example, ".TO" must be considered before ".T"). Persisted identities reject
# every suffix in this tuple because each one has a canonicalized spelling.
EXCHANGE_TICKER_SUFFIXES = (
  ".TO",
  ".HK",
  ".AX",
  ".PA",
  ".DE",
  ".SW",
  ".AS",
  ".SS",
  ".SZ",
  ".OL",
  ".MI",
  ".CO",
  ".ST",
  ".HE",
  ".BR",
  ".SA",
  ".SI",
  ".KS",
  ".TW",
  ".BO",
  ".NS",
  ".L",
  ".T",
)

# Dot and hyphen spellings are accepted only at normalization boundaries.
# Persisted CONTRACT tickers exclude hyphens by grammar and explicitly reject
# the dot aliases so BRK.A/BRK-A have the sole canonical spelling BRKA.
SHARE_CLASS_TICKER_SUFFIXES = (".A", ".B", "-A", "-B")
CANONICALIZING_TICKER_SUFFIXES = tuple(
  dict.fromkeys((
    *EXCHANGE_TICKER_SUFFIXES,
    *(
      suffix
      for suffix in SHARE_CLASS_TICKER_SUFFIXES
      if suffix.startswith(".")
    ),
  ))
)


def normalize_ticker(raw: str) -> str:
  """Normalize a boundary ticker spelling without accepting invalid syntax."""

  normalized = raw.strip().upper()
  if ".." in normalized:
    return normalized
  if normalized.endswith("."):
    normalized = normalized[:-1]

  for suffix in EXCHANGE_TICKER_SUFFIXES:
    if normalized.endswith(suffix):
      normalized = normalized[: -len(suffix)]
      break

  for suffix in SHARE_CLASS_TICKER_SUFFIXES:
    if normalized.endswith(suffix) and len(normalized) > len(suffix):
      normalized = normalized[: -len(suffix)] + suffix[-1]
      break

  return normalized


def is_contract_ticker(value: str) -> bool:
  """Return whether *value* satisfies the normalized CONTRACT rule."""

  return (
    EXTENDED_TICKER_RE.fullmatch(value) is not None
    and ".." not in value
    and not value.endswith(".")
  )


def normalize_contract_ticker(
  raw: object,
  *,
  field_name: str = "ticker",
) -> str:
  """Normalize a boundary value and require canonical CONTRACT syntax."""

  if raw is None or (isinstance(raw, str) and not raw.strip()):
    raise ValueError(f"{field_name} is required")
  if not isinstance(raw, str):
    raise ValueError(f"{field_name} must be a string")
  normalized = normalize_ticker(raw)
  if not normalized:
    raise ValueError(f"{field_name} is required")
  renormalized = normalize_ticker(normalized)
  if renormalized != normalized:
    raise ValueError(
      f"invalid {field_name} after normalization: {raw!r} -> {normalized!r}; "
      "ticker normalization must reach a fixed point and satisfy the "
      "CONTRACT rule"
    )
  if not is_contract_ticker(normalized):
    raise ValueError(
      f"invalid {field_name} after normalization: {raw!r} -> {normalized!r}; "
      "ticker must satisfy the CONTRACT rule"
    )
  return normalized


def is_entity_ticker(value: str) -> bool:
  """Return whether *value* is a bounded entity with at least one letter."""

  return bool(ENTITY_TICKER_RE.fullmatch(value)) and any(
    "A" <= char <= "Z" for char in value
  )


def is_canonical_contract_ticker(value: object) -> bool:
  """Return whether *value* is an exact canonical persisted ticker."""

  return (
    type(value) is str
    and EXTENDED_TICKER_RE.fullmatch(value) is not None
    and ".." not in value
    and not value.endswith(".")
    and not value.endswith(CANONICALIZING_TICKER_SUFFIXES)
  )


def require_canonical_contract_ticker(
  value: object,
  *,
  field_name: str = "ticker",
) -> str:
  """Require the sole persisted CONTRACT ticker spelling without coercion."""

  if type(value) is not str:
    raise ValueError(f"{field_name} must be an exact string")
  if not is_canonical_contract_ticker(value):
    raise ValueError(
      f"{field_name} must be a canonical persisted CONTRACT ticker"
    )
  return value


__all__ = [
  "CANONICALIZING_TICKER_SUFFIXES",
  "ENTITY_TICKER_RE",
  "EXCHANGE_TICKER_SUFFIXES",
  "EXTENDED_TICKER_RE",
  "SHARE_CLASS_TICKER_SUFFIXES",
  "TICKER_CONTRACT_SEMANTIC_MANIFEST",
  "TICKER_INPUT_CONTRACT",
  "TICKER_NORMALIZATION_ALGORITHM_ID",
  "is_canonical_contract_ticker",
  "is_contract_ticker",
  "is_entity_ticker",
  "normalize_contract_ticker",
  "normalize_ticker",
  "require_canonical_contract_ticker",
]
