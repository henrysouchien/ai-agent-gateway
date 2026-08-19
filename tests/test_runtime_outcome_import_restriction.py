"""``runtime_outcome=`` has exactly one non-test caller (D-B3-4, T3-I08).

``build_task_result`` cannot see its caller, so the "only the settlement
constructor mints a mechanical outcome" rule is enforced structurally: a
separate keyword plus this repository-wide restriction.  If a second product
module starts passing ``runtime_outcome=``, the derivation has escaped its one
owner and this test is the tripwire.
"""

from __future__ import annotations

import ast
from pathlib import Path


_KEYWORD = "runtime_outcome"

# The sole non-test caller: the one settlement constructor for run_agent
# children, workflow nodes, and forks.
_ALLOWED_CALLERS = {
  "packages/agent-gateway/agent_gateway/sub_agent_narrative_result.py",
}

# The definition site itself declares the parameter; it is not a caller.
_DEFINITION_SITE = (
  "packages/agent-gateway/agent_gateway/sub_agent_result_contract.py"
)

_SEARCH_ROOTS = ("api", "packages", "scripts")

# Build/packaging output is a copy of product source, not a second caller.
_NON_SOURCE_SEGMENTS = frozenset({
  "build",
  "dist",
  "node_modules",
  "site-packages",
  "__pycache__",
  ".venv",
})


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[3]


def _product_sources(root: Path):
  for name in _SEARCH_ROOTS:
    base = root / name
    if not base.is_dir():
      continue
    for path in sorted(base.rglob("*.py")):
      relative = path.relative_to(root)
      if _NON_SOURCE_SEGMENTS.intersection(relative.parts):
        continue
      posix = relative.as_posix()
      if "/tests/" in f"/{posix}" or path.name.startswith("test_"):
        continue
      yield posix, path


def test_runtime_outcome_keyword_has_exactly_one_product_caller() -> None:
  root = _repo_root()
  callers: list[str] = []
  for relative, path in _product_sources(root):
    source = path.read_text(encoding="utf-8")
    if _KEYWORD not in source:
      continue
    tree = ast.parse(source)
    passes_keyword = any(
      isinstance(node, ast.Call)
      and any(
        keyword.arg == _KEYWORD
        for keyword in node.keywords
      )
      for node in ast.walk(tree)
    )
    if passes_keyword:
      callers.append(relative)

  assert set(callers) == _ALLOWED_CALLERS


def test_runtime_outcome_is_declared_only_by_the_settlement_envelope() -> None:
  root = _repo_root()
  definitions: list[str] = []
  for relative, path in _product_sources(root):
    source = path.read_text(encoding="utf-8")
    if _KEYWORD not in source:
      continue
    tree = ast.parse(source)
    for node in ast.walk(tree):
      if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        continue
      argument_names = {
        argument.arg
        for argument in (*node.args.args, *node.args.kwonlyargs)
      }
      if _KEYWORD in argument_names:
        definitions.append(relative)

  assert set(definitions) == {_DEFINITION_SITE}
