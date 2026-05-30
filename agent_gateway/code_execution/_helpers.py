from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from ._config import CodeExecutionConfig

if TYPE_CHECKING:
  from ._backends import ExecutionBackend


OnOutputChunk = Callable[[str, str], None]

_STREAM_READER_LIMIT = 1_048_576
_MAX_CHUNK_EVENTS = 500
_MAX_CHUNK_BYTES = 4096
_CODE_EXECUTE_IMAGE_SUFFIXES = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
}
_CODE_EXECUTE_STDERR_NOISE = (
  "Matplotlib is building the font cache; this may take a moment.",
)
_SANDBOX_ENV_DENYLIST = frozenset(
  {
    "AGENT_API_USER_CLAIM_HMAC_KEY",
    "AGENT_API_CLAIM_TTL_SECONDS",
    "GATEWAY_RESOLVER_HMAC_KEY",
    "CHAT_API_SECRET",
    "CHAT_VALID_KEYS",
    "EXCEL_MCP_API_KEY",
    "ADDIN_DISPATCH_API_KEY",
    "RISK_API_KEY",
  }
)


def _error(code: str, message: str, details: Optional[Dict[str, Any]] = None) -> Tuple[None, Dict[str, Any]]:
  payload: Dict[str, Any] = {"code": code, "message": message}
  if details is not None:
    payload["details"] = details
  return None, payload


def _truncate_text(text: str, max_bytes: int) -> Tuple[str, bool]:
  encoded = text.encode("utf-8", errors="replace")
  if len(encoded) <= max_bytes:
    return text, False
  clipped = encoded[: max(0, max_bytes - 3)] + b"..."
  return clipped.decode("utf-8", errors="replace"), True


def _truncate_stdio(stdout: str, stderr: str, limit: int) -> Tuple[str, str, bool]:
  out = stdout.encode("utf-8", errors="replace")
  err = stderr.encode("utf-8", errors="replace")
  total = len(out) + len(err)
  if total <= limit:
    return stdout, stderr, False

  err_budget = min(len(err), int(limit * 0.6))
  out_budget = max(0, limit - err_budget)

  clipped_out = out[: max(0, out_budget - 3)] + (b"..." if len(out) > out_budget else b"")
  clipped_err = err[: max(0, err_budget - 3)] + (b"..." if len(err) > err_budget else b"")
  return (
    clipped_out.decode("utf-8", errors="replace"),
    clipped_err.decode("utf-8", errors="replace"),
    True,
  )


def _build_subprocess_env(extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
  env = os.environ.copy()
  for key, value in (extra_env or {}).items():
    if not key:
      continue
    text_value = str(value)
    existing = env.get(key)
    env[key] = f"{text_value}{os.pathsep}{existing}" if existing else text_value
  return _strip_sandbox_secrets(env)


def _strip_sandbox_secrets(env: Dict[str, str]) -> Dict[str, str]:
  for key in _SANDBOX_ENV_DENYLIST:
    env.pop(key, None)
  return env


def _default_code_execute_preamble(task_id: str = "") -> str:
  plot_prefix = f"_plot_{task_id}_" if task_id else "_plot_"
  return f"""\
import importlib.abc as _importlib_abc
import os as _os
import sys as _sys

_WORK_DIR = _os.getcwd()
_PLOT_PREFIX = {plot_prefix!r}
_PLOT_COUNTER = [0]
_os.environ.setdefault("MPLBACKEND", "Agg")


def _code_execute_patch_pyplot(_plt):
    if getattr(_plt, "_code_execute_show_patched", False):
        return

    def _patched_show(*args, **kwargs):
        for _fig_num in _plt.get_fignums():
            _fig = _plt.figure(_fig_num)
            _PLOT_COUNTER[0] += 1
            _fig.savefig(
                _os.path.join(_WORK_DIR, f"{{_PLOT_PREFIX}}{{_PLOT_COUNTER[0]}}.png"),
                dpi=150, bbox_inches="tight",
            )
        _plt.close("all")

    _plt.show = _patched_show
    _plt._code_execute_show_patched = True


class _CodeExecutePyplotLoader(_importlib_abc.Loader):
    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        create_module = getattr(self._loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        _code_execute_patch_pyplot(module)


class _CodeExecutePyplotFinder(_importlib_abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "matplotlib.pyplot":
            return None
        for finder in list(_sys.meta_path):
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is None:
                continue
            if spec.loader is not None:
                spec.loader = _CodeExecutePyplotLoader(spec.loader)
            return spec
        return None


_sys.meta_path.insert(0, _CodeExecutePyplotFinder())
if "matplotlib.pyplot" in _sys.modules:
    _code_execute_patch_pyplot(_sys.modules["matplotlib.pyplot"])
"""


def _build_code_execute_preamble(config: CodeExecutionConfig, task_id: str = "") -> str:
  if config.build_preamble is not None:
    return config.build_preamble(task_id)

  preamble = _default_code_execute_preamble(task_id).rstrip()
  suffix = (config.preamble_suffix or "").strip("\n")
  if suffix:
    preamble = f"{preamble}\n\n{suffix}"
  return preamble + "\n"


def _prepare_code_execute_env(config: CodeExecutionConfig) -> Dict[str, str]:
  env = _build_subprocess_env(config.extra_env)
  env["PYTHONUNBUFFERED"] = "1"

  shared_cache_root = Path(tempfile.gettempdir()) / "code_execute_cache"
  mpl_config_dir = shared_cache_root / "mplconfig"
  xdg_cache_dir = shared_cache_root / "xdg"
  mpl_config_dir.mkdir(parents=True, exist_ok=True)
  xdg_cache_dir.mkdir(parents=True, exist_ok=True)
  env.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
  env.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

  if config.prepare_env is not None:
    prepared = config.prepare_env(env)
    if prepared is not None:
      env = prepared
  return _strip_sandbox_secrets(env)


def _strip_code_execute_stderr_noise(stderr: str) -> str:
  if not stderr:
    return stderr
  lines = stderr.splitlines()
  filtered = [line for line in lines if line.strip() not in _CODE_EXECUTE_STDERR_NOISE]
  if stderr.endswith("\n") and filtered:
    return "\n".join(filtered) + "\n"
  return "\n".join(filtered)


def _snapshot_image_mtimes(work_dir: Path) -> Dict[Path, int]:
  mtimes: Dict[Path, int] = {}
  for candidate in work_dir.rglob("*"):
    if not candidate.is_file():
      continue
    if candidate.suffix.lower() not in _CODE_EXECUTE_IMAGE_SUFFIXES:
      continue
    try:
      mtimes[candidate.resolve()] = candidate.stat().st_mtime_ns
    except OSError:
      continue
  return mtimes


def _collect_generated_images(
  work_dir: Path,
  started_ns: int,
  config: CodeExecutionConfig,
  *,
  before_mtimes: Optional[Dict[Path, int]] = None,
  task_id: str | None = None,
) -> List[Dict[str, Any]]:
  images: List[Dict[str, Any]] = []
  captured_count = 0
  task_prefix = f"_plot_{task_id}_" if task_id else ""

  for candidate in sorted(work_dir.rglob("*")):
    if not candidate.is_file():
      continue
    media_type = _CODE_EXECUTE_IMAGE_SUFFIXES.get(candidate.suffix.lower())
    if media_type is None:
      continue
    if task_prefix and not candidate.name.startswith(task_prefix):
      continue
    try:
      stat = candidate.stat()
    except OSError as exc:
      images.append({"filename": candidate.name, "skipped": True, "reason": f"stat_failed: {exc}"})
      continue
    previous_mtime = (before_mtimes or {}).get(candidate.resolve())
    if stat.st_mtime_ns < started_ns and (previous_mtime is not None and stat.st_mtime_ns <= previous_mtime):
      continue
    if captured_count >= config.max_images:
      images.append({"filename": candidate.name, "skipped": True, "reason": f"exceeds {config.max_images} image limit"})
      continue
    try:
      data_base64 = base64.b64encode(candidate.read_bytes()).decode("ascii")
    except OSError as exc:
      images.append({"filename": candidate.name, "skipped": True, "reason": f"read_failed: {exc}"})
      continue
    if len(data_base64.encode("ascii")) > config.max_image_base64_bytes:
      images.append(
        {
          "filename": candidate.name,
          "skipped": True,
          "reason": f"exceeds {config.max_image_base64_bytes // 1024}KB",
        }
      )
      continue
    images.append(
      {
        "filename": candidate.name,
        "media_type": media_type,
        "data_base64": data_base64,
      }
    )
    captured_count += 1

  return images


def _task_script_path(work_dir: Path, task_id: str) -> Path:
  return work_dir / f"_task_{task_id}.py"


def _task_stdout_path(work_dir: Path, task_id: str) -> Path:
  return work_dir / f"_task_{task_id}_stdout.log"


def _task_stderr_path(work_dir: Path, task_id: str) -> Path:
  return work_dir / f"_task_{task_id}_stderr.log"


def _write_code_execute_script(
  work_dir: Path,
  code: str,
  config: CodeExecutionConfig,
  *,
  task_id: str = "",
) -> Path:
  if task_id:
    script_path = _task_script_path(work_dir, task_id)
    script_path.write_text(_build_code_execute_preamble(config, task_id) + "\n" + code, encoding="utf-8")
    return script_path

  with tempfile.NamedTemporaryFile(
    mode="w",
    encoding="utf-8",
    suffix=".py",
    prefix="_code_execute_",
    dir=str(work_dir),
    delete=False,
  ) as handle:
    handle.write(_build_code_execute_preamble(config))
    handle.write("\n")
    handle.write(code)
    return Path(handle.name)


def _remove_path(path: Path | None) -> None:
  if path is None:
    return
  try:
    path.unlink()
  except FileNotFoundError:
    pass
  except OSError:
    pass


def _remove_task_artifacts(work_dir: Path, task_id: str) -> None:
  patterns = [
    f"_task_{task_id}.py",
    f"_task_{task_id}_stdout.log",
    f"_task_{task_id}_stderr.log",
    f"_plot_{task_id}_*",
  ]
  for pattern in patterns:
    for candidate in work_dir.glob(pattern):
      _remove_path(candidate)


def _tail_file(path_value: str | Path, n: int = 20) -> str:
  path = Path(path_value)
  if not path.exists():
    return ""
  try:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(True)
  except OSError:
    return ""
  return "".join(lines[-n:])


def _read_truncated(path_value: str | Path, max_bytes: int) -> Tuple[str, bool]:
  path = Path(path_value)
  if not path.exists():
    return "", False
  try:
    with path.open("rb") as handle:
      data = handle.read(max_bytes + 1)
  except OSError:
    return "", False
  truncated = len(data) > max_bytes
  return data[:max_bytes].decode("utf-8", errors="replace"), truncated


async def code_execute(
  tool_input: Dict[str, Any],
  *,
  session_work_dir: Optional[str] = None,
  on_output: Optional[OnOutputChunk] = None,
  backend: Optional["ExecutionBackend"] = None,
  config: CodeExecutionConfig | None = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
  cfg = config or CodeExecutionConfig()

  code = str(tool_input.get("code") or "")
  if not code.strip():
    return _error("invalid_input", "code is required")

  timeout_ms_raw = tool_input.get("timeout_ms", cfg.default_timeout_ms)
  try:
    timeout_ms = int(timeout_ms_raw)
  except (TypeError, ValueError):
    return _error("invalid_input", "timeout_ms must be an integer")
  timeout_ms = max(1000, min(timeout_ms, cfg.max_timeout_ms))

  work_dir_path = Path(session_work_dir or os.getcwd()).expanduser().resolve()
  try:
    work_dir_path.mkdir(parents=True, exist_ok=True)
  except OSError as exc:
    return _error("write_failed", f"Failed to create work directory: {exc}")
  if not work_dir_path.is_dir():
    return _error("not_found", f"Directory not found: {work_dir_path}")
  if backend is None:
    return _error("internal_error", "Execution backend is required")

  try:
    env = _prepare_code_execute_env(cfg)

    chunk_count = [0]

    def _emit_chunk(stream_name: str, text: str) -> None:
      if on_output is None:
        return
      chunk_count[0] += 1
      if chunk_count[0] > _MAX_CHUNK_EVENTS:
        return
      encoded = text.encode("utf-8")
      if len(encoded) > _MAX_CHUNK_BYTES:
        chunk_text = encoded[:_MAX_CHUNK_BYTES].decode("utf-8", errors="ignore")
      else:
        chunk_text = text
      on_output(stream_name, chunk_text)

    result = await backend.execute(
      code,
      str(work_dir_path),
      timeout_ms=timeout_ms,
      env=env,
      on_output=_emit_chunk if on_output is not None else None,
    )
    stdout = str(result.get("stdout") or "")
    stderr = _strip_code_execute_stderr_noise(str(result.get("stderr") or ""))
    stdout, stderr, truncated = _truncate_stdio(stdout, stderr, cfg.max_output_bytes)
    return {
      "stdout": stdout,
      "stderr": stderr,
      "return_code": result.get("return_code"),
      "images": result.get("images") or [],
      "timed_out": bool(result.get("timed_out")),
      "duration_ms": result.get("duration_ms"),
      "truncated": bool(result.get("truncated")) or truncated,
    }, None
  except Exception as exc:
    return _error("tool_error", f"code_execute failed: {exc}")
