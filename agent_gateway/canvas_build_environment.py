"""Pinned, hermetic Canvas TSX build environment."""

from __future__ import annotations

from dataclasses import dataclass
import asyncio
from importlib.resources import files
import json
import logging
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import tempfile
import threading
from typing import Any, Mapping

from . import canvas_kit_contract


log = logging.getLogger(__name__)


CANVAS_BUILD_DIR_ENV = "CANVAS_BUILD_DIR"
BUILD_TIMEOUT_SECONDS = 20.0
BUILD_MEMORY_BYTES = 768 * 1024 * 1024
BUILD_OUTPUT_MAX_BYTES = 128 * 1024
MAX_DIAGNOSTICS = 8
CANVAS_RUNTIME_SUPPORT_FILES = (
  ".node-version",
  "node_checksums.json",
  "package.json",
  "package-lock.json",
  "build.mjs",
  "policy.mjs",
  "tsconfig.json",
)
_ALLOWED_IMPORTS = frozenset({"react", "recharts", "@hank/canvas-kit"})
_FORBIDDEN_PATTERNS = (
  r"\bfetch\b", r"\bXMLHttpRequest\b", r"\bWebSocket\b", r"\bEventSource\b",
  r"\bimportScripts\b", r"\beval\b", r"\bFunction\b", r"\bdocument\s*\.\s*cookie\b",
  r"\blocalStorage\b", r"\bsessionStorage\b", r"\bindexedDB\b", r"\bimport\s*\(",
  r"\bsetTimeout\b", r"\bsetInterval\b", r"\bpostMessage\b",
)


@dataclass(frozen=True)
class CanvasBuildPreflight:
  build_dir: Path
  node: Path
  tsc: Path
  esbuild: Path
  toolchain_version: str


@dataclass(frozen=True)
class _CanvasBuildPreflightConfig:
  build_dir: str
  node_binary: str
  subprocess_env: tuple[tuple[str, str], ...]


_CANVAS_BUILD_PREFLIGHT_CACHE: dict[
  _CanvasBuildPreflightConfig, CanvasBuildPreflight | None,
] = {}
_CANVAS_BUILD_PREFLIGHT_LOCK = threading.Lock()


class CanvasBuildFailure(Exception):
  def __init__(self, stage: str, diagnostics: list[dict[str, Any]]):
    super().__init__(diagnostics[0]["message"] if diagnostics else stage)
    self.stage = stage
    self.diagnostics = diagnostics[:MAX_DIAGNOSTICS]

  def payload(self) -> dict[str, Any]:
    return {"validation_failed": {"stage": self.stage, "diagnostics": self.diagnostics}}


def canvas_build_enabled(env: Mapping[str, str] | None = None) -> bool:
  values = os.environ if env is None else env
  return bool(str(values.get(CANVAS_BUILD_DIR_ENV, "")).strip())


def packaged_build_directory() -> Path:
  return Path(str(files("agent_gateway") / "canvas_build"))


def _verify_runtime_support_files(*, build_dir: Path, package_dir: Path) -> None:
  mismatches: list[str] = []
  for relative_path in CANVAS_RUNTIME_SUPPORT_FILES:
    canonical = package_dir / relative_path
    runtime = build_dir / relative_path
    if runtime.is_symlink() or not runtime.is_file():
      mismatches.append(f"{relative_path}: missing or not a regular file")
      continue
    try:
      matches = runtime.read_bytes() == canonical.read_bytes()
    except OSError as exc:
      mismatches.append(f"{relative_path}: {exc}")
      continue
    if not matches:
      mismatches.append(f"{relative_path}: differs from packaged canonical file")
  if mismatches:
    raise RuntimeError(
      "Canvas runtime support-file verification failed: " + "; ".join(mismatches)
    )


def _diagnostic(
  code: str,
  message: str,
  repair_hint: str,
  *,
  source: str | None = None,
  offset: int | None = None,
) -> dict[str, Any]:
  item: dict[str, Any] = {"code": code, "message": message, "repair_hint": repair_hint}
  if source is not None and offset is not None:
    item["line"] = source.count("\n", 0, offset) + 1
    line_start = source.rfind("\n", 0, offset) + 1
    item["column"] = offset - line_start + 1
  return item


def _run_version(argv: list[str], *, env: dict[str, str]) -> str:
  completed = subprocess.run(
    argv, check=False, capture_output=True, text=True, timeout=5, env=env,
  )
  if completed.returncode:
    raise RuntimeError((completed.stderr or completed.stdout).strip() or "command failed")
  return completed.stdout.strip()


def _validate_canvas_build_environment(
  config: _CanvasBuildPreflightConfig,
) -> CanvasBuildPreflight:
  unresolved_build_dir = Path(config.build_dir)
  if unresolved_build_dir.is_symlink():
    raise RuntimeError(f"CANVAS_BUILD_DIR must not be a symlink: {unresolved_build_dir}")
  build_dir = unresolved_build_dir.resolve()
  if not build_dir.is_dir():
    raise RuntimeError(f"CANVAS_BUILD_DIR is not a directory: {build_dir}")
  package_dir = packaged_build_directory().resolve()
  _verify_runtime_support_files(build_dir=build_dir, package_dir=package_dir)
  node = Path(config.node_binary)
  tsc = build_dir / "node_modules" / ".bin" / "tsc"
  esbuild = build_dir / "node_modules" / ".bin" / "esbuild"
  for name, path in (("tsc", tsc), ("esbuild", esbuild)):
    if not path.is_file():
      raise RuntimeError(f"Canvas {name} is absent: {path}; run canvas_build/provision.sh")
  expected_node = (package_dir / ".node-version").read_text(encoding="utf-8").strip()
  expected_versions = canvas_kit_contract.pinned_versions()
  clean_env = dict(config.subprocess_env)
  actual_node = _run_version([str(node), "--version"], env=clean_env).removeprefix("v")
  actual_tsc = _run_version([str(tsc), "--version"], env=clean_env).removeprefix("Version ")
  actual_esbuild = _run_version([str(esbuild), "--version"], env=clean_env)
  expected = (expected_node, expected_versions["typescript"], expected_versions["esbuild"])
  actual = (actual_node, actual_tsc, actual_esbuild)
  if actual != expected:
    raise RuntimeError(f"Canvas toolchain mismatch: expected {expected}, got {actual}")
  return CanvasBuildPreflight(
    build_dir=build_dir,
    node=node,
    tsc=tsc,
    esbuild=esbuild,
    toolchain_version=f"node/{actual_node} tsc/{actual_tsc} esbuild/{actual_esbuild}",
  )


def preflight_canvas_build_environment(
  env: Mapping[str, str] | None = None,
) -> CanvasBuildPreflight | None:
  """Return the process-owned Canvas capability snapshot for this configuration."""

  values = os.environ if env is None else env
  raw_dir = str(values.get(CANVAS_BUILD_DIR_ENV, "")).strip()
  if not raw_dir:
    return None
  node = Path(str(values.get("CANVAS_NODE_BINARY", "node")))
  config = _CanvasBuildPreflightConfig(
    build_dir=str(Path(raw_dir).expanduser()),
    node_binary=str(node),
    subprocess_env=tuple(sorted(sanitized_subprocess_env(values, node_binary=node).items())),
  )
  with _CANVAS_BUILD_PREFLIGHT_LOCK:
    if config in _CANVAS_BUILD_PREFLIGHT_CACHE:
      return _CANVAS_BUILD_PREFLIGHT_CACHE[config]
    try:
      preflight = _validate_canvas_build_environment(config)
    except subprocess.TimeoutExpired as exc:
      log.warning(
        "Canvas toolchain preflight timed out; Canvas artifacts are disabled "
        "for this process configuration: %s",
        exc,
      )
      preflight = None
    _CANVAS_BUILD_PREFLIGHT_CACHE[config] = preflight
    return preflight


def sanitized_subprocess_env(
  env: Mapping[str, str] | None = None,
  *,
  node_binary: Path | None = None,
) -> dict[str, str]:
  values = os.environ if env is None else env
  clean = {
    key: value for key, value in values.items()
    if key in {"PATH", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT"}
    and not key.lower().startswith("npm_config_")
  }
  clean.pop("NODE_PATH", None)
  clean.pop("NODE_OPTIONS", None)
  if node_binary and node_binary.parent != Path("."):
    clean["PATH"] = f"{node_binary.parent}{os.pathsep}{clean.get('PATH', '')}"
  clean["NO_PROXY"] = "*"
  clean["no_proxy"] = "*"
  return clean


def _masked_source(source: str) -> str:
  """Mask comments and strings while retaining offsets/newlines for policy positions."""

  chars = list(source)
  index = 0
  while index < len(chars):
    if source.startswith("//", index):
      end = source.find("\n", index)
      end = len(chars) if end < 0 else end
      for pos in range(index, end):
        chars[pos] = " "
      index = end
    elif source.startswith("/*", index):
      end = source.find("*/", index + 2)
      end = len(chars) - 2 if end < 0 else end
      for pos in range(index, min(len(chars), end + 2)):
        if chars[pos] != "\n":
          chars[pos] = " "
      index = end + 2
    elif chars[index] in {'"', "'", "`"}:
      quote = chars[index]
      chars[index] = " "
      index += 1
      while index < len(chars):
        if source[index] == "\\":
          chars[index] = " "
          if index + 1 < len(chars):
            chars[index + 1] = " "
          index += 2
          continue
        if source[index] == quote:
          chars[index] = " "
          index += 1
          break
        if chars[index] != "\n":
          chars[index] = " "
        index += 1
    else:
      index += 1
  return "".join(chars)


def check_module_policy(source: str) -> None:
  """Reject imports and executable vocabulary before invoking the toolchain."""

  diagnostics: list[dict[str, Any]] = []
  import_pattern = re.compile(r"\b(?:import|export)\s+(?:[^;]*?\s+from\s+)?[\"']([^\"']+)[\"']")
  for match in import_pattern.finditer(source):
    if match.group(1) not in _ALLOWED_IMPORTS:
      diagnostics.append(_diagnostic(
        "import_allowlist", f"Import {match.group(1)!r} is not allowed.",
        "Import only react, recharts, or @hank/canvas-kit.", source=source, offset=match.start(),
      ))
  masked = _masked_source(source)
  for pattern in _FORBIDDEN_PATTERNS:
    match = re.search(pattern, masked)
    if match:
      diagnostics.append(_diagnostic(
        "forbidden_identifier", f"Forbidden Canvas identifier: {match.group(0).strip()}.",
        "Remove browser, timer, storage, dynamic-code, and messaging APIs.",
        source=source, offset=match.start(),
      ))
  defaults = list(re.finditer(r"\bexport\s+default\b", masked))
  if len(defaults) != 1:
    diagnostics.append(_diagnostic(
      "default_export_count", "Canvas source must contain exactly one default export.",
      "Export one React component as default.", source=source, offset=defaults[0].start() if defaults else 0,
    ))
  # At brace depth zero, executable expression statements are side effects. Imports,
  # declarations, and the single default component are the only allowed statement forms.
  depth = 0
  start = 0
  statements: list[tuple[int, str]] = []
  for index, char in enumerate(masked):
    if char in "{([":
      depth += 1
    elif char in "})]":
      depth = max(0, depth - 1)
    elif char == ";" and depth == 0:
      statements.append((start, masked[start:index + 1].strip()))
      start = index + 1
  tail = masked[start:].strip()
  if tail:
    statements.append((start, tail))
  allowed_start = re.compile(
    r"^(?:import\b|export\s+(?:default\s+)?(?:function|class|const|type|interface)\b|"
    r"export\s*\{|const\b|let\b|type\b|interface\b|function\b|class\b)"
  )
  for offset, statement in statements:
    if statement and not allowed_start.match(statement):
      real = offset + len(masked[offset:]) - len(masked[offset:].lstrip())
      diagnostics.append(_diagnostic(
        "module_scope_side_effect", "Executable statement is not allowed at module scope.",
        "Move computation into the component and keep module scope to imports, types, constants, and declarations.",
        source=source, offset=real,
      ))
      break
  if diagnostics:
    raise CanvasBuildFailure("module_policy", diagnostics)


def _limit_resources() -> None:
  memory_limit = resource.RLIMIT_DATA if sys.platform == "darwin" else resource.RLIMIT_AS
  resource.setrlimit(memory_limit, (BUILD_MEMORY_BYTES, BUILD_MEMORY_BYTES))
  resource.setrlimit(resource.RLIMIT_FSIZE, (BUILD_OUTPUT_MAX_BYTES * 8, BUILD_OUTPUT_MAX_BYTES * 8))


def _run_build_command(
  argv: list[str], *, cwd: Path, env: dict[str, str], stage: str,
) -> subprocess.CompletedProcess[bytes]:
  try:
    completed = subprocess.run(
      argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, capture_output=True,
      timeout=BUILD_TIMEOUT_SECONDS, check=False,
      # Darwin's posix_spawn bridge rejects these rlimits; production Linux
      # applies both memory and file ceilings. Timeout/output caps apply everywhere.
      preexec_fn=_limit_resources if os.name == "posix" and sys.platform != "darwin" else None,
    )
  except subprocess.TimeoutExpired as exc:
    raise CanvasBuildFailure("build_environment", [_diagnostic(
      "toolchain_timeout", f"Canvas {stage} exceeded {BUILD_TIMEOUT_SECONDS:g}s.",
      "Reduce source complexity and retry; persistent timeouts require operator action.",
    )]) from exc
  output = completed.stdout + completed.stderr
  if len(output) > BUILD_OUTPUT_MAX_BYTES:
    raise CanvasBuildFailure("build_environment", [_diagnostic(
      "toolchain_output_limit", "Canvas toolchain output exceeded its safety limit.",
      "Fix the first reported errors or reduce generated source complexity.",
    )])
  return completed


async def _run_build_command_async(
  argv: list[str], *, cwd: Path, env: dict[str, str], stage: str,
) -> tuple[int, bytes, bytes]:
  process = await asyncio.create_subprocess_exec(
    *argv, cwd=cwd, env=env, stdin=asyncio.subprocess.DEVNULL,
    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    preexec_fn=_limit_resources if os.name == "posix" and sys.platform != "darwin" else None,
  )
  try:
    stdout, stderr = await asyncio.wait_for(process.communicate(), BUILD_TIMEOUT_SECONDS)
  except asyncio.CancelledError:
    process.kill()
    await process.communicate()
    raise
  except TimeoutError as exc:
    process.kill()
    await process.communicate()
    raise CanvasBuildFailure("build_environment", [_diagnostic(
      "toolchain_timeout", f"Canvas {stage} exceeded {BUILD_TIMEOUT_SECONDS:g}s.",
      "Reduce source complexity and retry; persistent timeouts require operator action.",
    )]) from exc
  if len(stdout) + len(stderr) > BUILD_OUTPUT_MAX_BYTES:
    raise CanvasBuildFailure("build_environment", [_diagnostic(
      "toolchain_output_limit", "Canvas toolchain output exceeded its safety limit.",
      "Fix the first reported errors or reduce generated source complexity.",
    )])
  return process.returncode or 0, stdout, stderr


_TSC_DIAGNOSTIC_RE = re.compile(r"\((\d+),(\d+)\): error (TS\d+): (.+)")


def _typecheck_diagnostics(raw: str, source: str) -> list[dict[str, Any]]:
  diagnostics: list[dict[str, Any]] = []
  for output_line in raw.splitlines():
    match = _TSC_DIAGNOSTIC_RE.search(output_line)
    if match:
      line = int(match.group(1))
      diagnostics.append({
        "line": line,
        "column": int(match.group(2)),
        "code": match.group(3),
        "message": match.group(4),
        "repair_hint": canvas_kit_contract.component_repair_hint(
          source,
          line,
          "Correct the referenced type or component prop and retry.",
        ),
      })
  return diagnostics


def build_canvas_bundle(source: str, preflight: CanvasBuildPreflight) -> bytes:
  """Typecheck and bundle one source in an isolated, automatically-cleaned directory."""

  check_module_policy(source)
  contract_dir = canvas_kit_contract.packaged_contract_directory()
  config_template = json.loads((contract_dir / "tsconfig.typecheck.json").read_text())
  with tempfile.TemporaryDirectory(prefix="hank-canvas-build-") as raw_temp:
    temp_dir = Path(raw_temp)
    input_path = temp_dir / "source.tsx"
    bundle_path = temp_dir / "bundle.js"
    input_path.write_text(source, encoding="utf-8")
    options = dict(config_template["compilerOptions"])
    options["baseUrl"] = str(contract_dir)
    options["typeRoots"] = [str(contract_dir / "types" / "node_modules" / "@types")]
    options["paths"] = {
      key: [str(contract_dir / value[0].removeprefix("./"))]
      for key, value in options["paths"].items()
    }
    config_path = temp_dir / "tsconfig.json"
    config_path.write_text(json.dumps({"compilerOptions": options, "files": [str(input_path)]}))
    clean_env = sanitized_subprocess_env(node_binary=preflight.node)
    policy = _run_build_command(
      [str(preflight.node), str(preflight.build_dir / "policy.mjs"), str(input_path)],
      cwd=temp_dir, env=clean_env, stage="module policy",
    )
    if policy.returncode:
      try:
        diagnostics = json.loads(policy.stdout.decode("utf-8"))
      except (UnicodeError, json.JSONDecodeError):
        diagnostics = []
      raise CanvasBuildFailure("module_policy", diagnostics or [_diagnostic(
        "module_policy_failed", "Canvas AST module policy validation failed.",
        "Use one default component and remove executable module-scope code.",
      )])
    checked = _run_build_command(
      [str(preflight.tsc), "--project", str(config_path), "--pretty", "false"],
      cwd=temp_dir, env=clean_env, stage="typecheck",
    )
    if checked.returncode:
      raw = (checked.stdout + checked.stderr).decode("utf-8", "replace")
      diagnostics = _typecheck_diagnostics(raw, source)
      raise CanvasBuildFailure("typecheck", diagnostics or [_diagnostic(
        "typecheck_failed", "Canvas TypeScript validation failed.",
        "Correct the first TypeScript error and retry.",
      )])
    built = _run_build_command(
      [str(preflight.node), str(preflight.build_dir / "build.mjs"),
       str(input_path), str(bundle_path), str(contract_dir)],
      cwd=temp_dir, env=clean_env, stage="bundle",
    )
    if built.returncode or not bundle_path.is_file():
      message = (built.stderr or built.stdout).decode("utf-8", "replace").strip()
      raise CanvasBuildFailure("bundle", [_diagnostic(
        "bundle_failed", message[:2000] or "Canvas bundling failed.",
        "Correct unsupported syntax or module structure and retry.",
      )])
    return bundle_path.read_bytes()


async def build_canvas_bundle_async(source: str, preflight: CanvasBuildPreflight) -> bytes:
  """Cancellation-safe async variant used by live tool handlers."""

  check_module_policy(source)
  contract_dir = canvas_kit_contract.packaged_contract_directory()
  config_template = json.loads((contract_dir / "tsconfig.typecheck.json").read_text())
  with tempfile.TemporaryDirectory(prefix="hank-canvas-build-") as raw_temp:
    temp_dir = Path(raw_temp)
    input_path = temp_dir / "source.tsx"
    bundle_path = temp_dir / "bundle.js"
    input_path.write_text(source, encoding="utf-8")
    options = dict(config_template["compilerOptions"])
    options["baseUrl"] = str(contract_dir)
    options["typeRoots"] = [str(contract_dir / "types" / "node_modules" / "@types")]
    options["paths"] = {
      key: [str(contract_dir / value[0].removeprefix("./"))]
      for key, value in options["paths"].items()
    }
    config_path = temp_dir / "tsconfig.json"
    config_path.write_text(json.dumps({"compilerOptions": options, "files": [str(input_path)]}))
    clean_env = sanitized_subprocess_env(node_binary=preflight.node)
    returncode, stdout, _stderr = await _run_build_command_async(
      [str(preflight.node), str(preflight.build_dir / "policy.mjs"), str(input_path)],
      cwd=temp_dir, env=clean_env, stage="module policy",
    )
    if returncode:
      try:
        diagnostics = json.loads(stdout.decode("utf-8"))
      except (UnicodeError, json.JSONDecodeError):
        diagnostics = []
      raise CanvasBuildFailure("module_policy", diagnostics or [_diagnostic(
        "module_policy_failed", "Canvas AST module policy validation failed.",
        "Use one default component and remove executable module-scope code.",
      )])
    returncode, stdout, stderr = await _run_build_command_async(
      [str(preflight.tsc), "--project", str(config_path), "--pretty", "false"],
      cwd=temp_dir, env=clean_env, stage="typecheck",
    )
    if returncode:
      raw = (stdout + stderr).decode("utf-8", "replace")
      diagnostics = _typecheck_diagnostics(raw, source)
      raise CanvasBuildFailure("typecheck", diagnostics or [_diagnostic(
        "typecheck_failed", "Canvas TypeScript validation failed.",
        "Correct the first TypeScript error and retry.",
      )])
    returncode, stdout, stderr = await _run_build_command_async(
      [str(preflight.node), str(preflight.build_dir / "build.mjs"),
       str(input_path), str(bundle_path), str(contract_dir)],
      cwd=temp_dir, env=clean_env, stage="bundle",
    )
    if returncode or not bundle_path.is_file():
      message = (stderr or stdout).decode("utf-8", "replace").strip()
      raise CanvasBuildFailure("bundle", [_diagnostic(
        "bundle_failed", message[:2000] or "Canvas bundling failed.",
        "Correct unsupported syntax or module structure and retry.",
      )])
    return bundle_path.read_bytes()


__all__ = [
  "CANVAS_BUILD_DIR_ENV", "CANVAS_RUNTIME_SUPPORT_FILES",
  "CanvasBuildFailure", "CanvasBuildPreflight",
  "build_canvas_bundle", "build_canvas_bundle_async", "canvas_build_enabled", "check_module_policy",
  "packaged_build_directory", "preflight_canvas_build_environment",
  "sanitized_subprocess_env",
]
