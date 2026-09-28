"""Sibling MCP-server environment delivery for a contained autonomous child.

A contained child may not read another application's credential file: the
credential class is denied to it by policy, so a sibling server started inside
containment loses the configuration it reads for itself when the uncontained
gateway starts it. The parent can read that file, so it reads the dotenv each
declared server would have read from its own ``cwd`` and hands the values to
the child in the launch environment -- the same channel the child's provider
keys already travel.

The file is derived from the declared ``cwd`` in the MCP config template (the
template owns which servers exist and where they run); nothing here holds a
path. Only a *sibling* application's file is read: a server declared on this
gateway's own interpreter runs inside this application, whose configuration
reaches a child only through the gateway's positive projection. What a sibling
file may supply is decided from owners, not from a list of names: the child's
runtime contract, the gateway's own environment, the shape of the name
(credential and identity material is authority), and the template's own env
block. A secret the gateway gains tomorrow is therefore withheld by default.
Values are never logged.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values


AUTONOMOUS_CHILD_SERVER_DOTENV_ENV = "AUTONOMOUS_MCP_SERVER_DOTENV"

_TEMPLATE_ENV = "MCP_CONFIG_TEMPLATE"
_ENV_REF_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
_OWN_INTERPRETER = Path(sys.executable).resolve()
_OWN_INTERPRETER_PREFIX = Path(sys.prefix).resolve()
# The gateway's virtualenv lives inside the application it serves, so its
# parent is the root a server declared on this application would run from.
_OWN_APPLICATION_ROOT = _OWN_INTERPRETER_PREFIX.parent
# Authority is recognized by what a name *is*, not by a list of the names that
# happen to exist today: the trailing segment of an environment variable name
# is the repository-wide convention for what its value holds. A sibling file
# may supply operational configuration (`SCHWAB_ENABLED`, `SCHWAB_TOKEN_PATH`,
# `CORPUS_ROOT`) but never credential or identity material, so a secret this
# gateway gains tomorrow is withheld without anyone editing this module.
_AUTHORITY_NAME_SEGMENTS = frozenset({
  "AUTH",
  "BYPASS",
  "CHAT_ID",
  "CREDENTIAL",
  "CREDENTIALS",
  "EMAIL",
  "KEY",
  "KEYS",
  "KEY_ID",
  "KEY_IDS",
  "PASSPHRASE",
  "PASSWORD",
  "PEPPER",
  "PEPPERS",
  "SECRET",
  "SECRETS",
  "SEED",
  "SIGNATURE",
  "TOKEN",
  "TOKENS",
  "USER_ID",
  "USER_IDS",
  "USER_SLUG",
})
# A serialization suffix names the encoding, not the value: the segment in
# front of it is what the variable holds.
_ENCODING_NAME_SEGMENTS = frozenset({
  "B64",
  "BASE64",
  "HEX",
  "JSON",
  "PEM",
})
_MAX_AUTHORITY_SEGMENT_WORDS = max(
  len(segment.split("_")) for segment in _AUTHORITY_NAME_SEGMENTS
)


def _expanded(raw: object, environ: Mapping[str, str]) -> str | None:
  """Substitute ``${VAR}`` references from the child's environment."""

  text = str(raw or "").strip()
  if not text:
    return None
  unresolved = False

  def _replace(match: "re.Match[str]") -> str:
    nonlocal unresolved
    value = str(environ.get(match.group(1)) or "").strip()
    if not value:
      unresolved = True
      return ""
    return value

  expanded = _ENV_REF_RE.sub(_replace, text).strip()
  if unresolved or not expanded:
    return None
  return expanded


def _expanded_server_cwd(raw: object, environ: Mapping[str, str]) -> Path | None:
  """Resolve a declared ``cwd`` against the child's environment."""

  expanded = _expanded(raw, environ)
  if expanded is None:
    return None
  path = Path(expanded).expanduser()
  # A relative cwd resolves against the child's working directory, which is not
  # the declaring application's root; only an absolute root names a real file.
  return path if path.is_absolute() else None


def _belongs_to_this_application(
  command: object,
  cwd: Path,
  environ: Mapping[str, str],
) -> bool:
  """Whether a declared server runs inside this gateway's own application."""

  if cwd.resolve() == _OWN_APPLICATION_ROOT:
    return True
  expanded = _expanded(command, environ)
  if expanded is None:
    return False
  interpreter = Path(expanded).expanduser()
  if not interpreter.is_absolute():
    return False
  resolved = interpreter.resolve()
  return (
    resolved == _OWN_INTERPRETER
    or _OWN_INTERPRETER_PREFIX in resolved.parents
  )


def _is_authority_material(name: str) -> bool:
  """Whether a variable name holds credential or identity material."""

  words = [word for word in name.split("_") if word]
  while words and words[-1] in _ENCODING_NAME_SEGMENTS:
    words.pop()
  return any(
    "_".join(words[-count:]) in _AUTHORITY_NAME_SEGMENTS
    for count in range(1, _MAX_AUTHORITY_SEGMENT_WORDS + 1)
    if count <= len(words)
  )


def _declared_env_references(server_config: Mapping[str, object]) -> frozenset[str]:
  """Names a server's own env block takes from this environment.

  The MCP config template owns what each server receives, so a name it
  declares is this gateway's own statement that the server runs on it -- the
  seven Risk servers declare ``${GATEWAY_API_KEY}`` and the production
  template declares their AWS principal. Nothing a sibling file holds can
  change that declaration.
  """

  env_block = server_config.get("env")
  if not isinstance(env_block, dict):
    return frozenset()
  return frozenset(
    reference
    for value in env_block.values()
    if isinstance(value, str)
    for reference in _ENV_REF_RE.findall(value)
  )


def server_dotenv_payload(
  environ: Mapping[str, str],
  *,
  gateway_environ: Mapping[str, str],
  withheld: frozenset[str] = frozenset(),
) -> str | None:
  """Read each sibling server's own dotenv into the child's launch value.

  ``environ`` is the environment the child will receive: it names the template
  and carries the values the gateway already forwards. ``gateway_environ`` is
  the gateway's own environment, which is the owner of every name this
  application is configured with -- a sibling never legitimately supplies one
  of those, whether or not the child's contract projects it.

  A sibling server receives the file it would have read for itself, minus:
  ``withheld`` (the child's runtime contract vocabulary -- the names the
  launcher computes, strips, or fences for the child itself), anything the
  child already carries or the gateway's own configuration owns, and anything
  whose name holds credential or identity material. That last rule is what
  keeps a secret out of the child without a list to maintain: authority is
  recognized from the name, so a gateway key added tomorrow is withheld by
  default rather than delivered until someone notices.

  A name the server's own env block declares is exempt from the last two
  rules: the template is the owner of what each server receives, and the seven
  Risk servers cannot reach this gateway without the ``${GATEWAY_API_KEY}``
  they declare. Everything else in that file is the declaring application's
  own operational configuration, which the gateway has no opinion about and
  the server stops working without.
  """

  template = str(environ.get(_TEMPLATE_ENV) or "").strip()
  if not template:
    return None
  try:
    template_data = json.loads(Path(template).read_text(encoding="utf-8"))
  except (OSError, UnicodeDecodeError, ValueError):
    return None
  servers = (
    template_data.get("mcpServers") if isinstance(template_data, dict) else None
  )
  if not isinstance(servers, dict):
    return None

  payload: dict[str, dict[str, str]] = {}
  for server_name, server_config in servers.items():
    if not isinstance(server_config, dict):
      continue
    cwd = _expanded_server_cwd(server_config.get("cwd"), environ)
    if cwd is None:
      continue
    if _belongs_to_this_application(server_config.get("command"), cwd, environ):
      # Not a sibling: this server runs inside the gateway's own application,
      # so the dotenv beside its cwd is this application's own credential
      # file, which no server reads for itself -- the gateway hands a server
      # it starts uncontained only that server's declared env block.
      continue
    try:
      values = dotenv_values(cwd / ".env")
    except (OSError, UnicodeDecodeError):
      continue
    declared = _declared_env_references(server_config)
    delivered = {
      str(name): str(value)
      for name, value in values.items()
      if isinstance(name, str)
      and isinstance(value, str)
      and value.strip()
      and name not in withheld
      and not str(environ.get(name) or "").strip()
      and (
        name in declared
        or (
          not str(gateway_environ.get(name) or "").strip()
          and not _is_authority_material(name)
        )
      )
    }
    if delivered:
      payload[str(server_name)] = delivered
  if not payload:
    return None
  return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def decode_server_dotenv(raw: str | None) -> dict[str, dict[str, str]]:
  """Decode the launch value into per-server dotenv values."""

  text = str(raw or "").strip()
  if not text:
    return {}
  try:
    data = json.loads(text)
  except ValueError:
    return {}
  if not isinstance(data, dict):
    return {}
  decoded: dict[str, dict[str, str]] = {}
  for server_name, values in data.items():
    if not isinstance(server_name, str) or not isinstance(values, dict):
      continue
    entries = {
      str(name): str(value)
      for name, value in values.items()
      if isinstance(name, str) and isinstance(value, str) and value.strip()
    }
    if entries:
      decoded[server_name] = entries
  return decoded


__all__ = [
  "AUTONOMOUS_CHILD_SERVER_DOTENV_ENV",
  "decode_server_dotenv",
  "server_dotenv_payload",
]
