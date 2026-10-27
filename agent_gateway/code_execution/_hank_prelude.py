"""Dependency-free pyplot capture for the persistent kernel runner.

`plt.show()` saves every open figure as `<plot_prefix><n>.png` in the work
directory and closes them. The meta-path finder is installed once per process;
the prefix and work directory are rebound per cell by `set_plot_prefix`.
"""

from __future__ import annotations

import importlib.abc as _importlib_abc
import os as _os
import sys as _sys

_state = {"work_dir": _os.getcwd(), "prefix": "_plot_", "counter": 0}
_os.environ.setdefault("MPLBACKEND", "Agg")


def _code_execute_patch_pyplot(_plt):
  if getattr(_plt, "_code_execute_show_patched", False):
    return

  def _patched_show(*args, **kwargs):
    for _fig_num in _plt.get_fignums():
      _fig = _plt.figure(_fig_num)
      _state["counter"] += 1
      _fig.savefig(
        _os.path.join(_state["work_dir"], f"{_state['prefix']}{_state['counter']}.png"),
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


def install_plot_capture() -> None:
  """Install the pyplot finder once; patch pyplot if it is already imported."""
  if not any(isinstance(finder, _CodeExecutePyplotFinder) for finder in _sys.meta_path):
    _sys.meta_path.insert(0, _CodeExecutePyplotFinder())
  if "matplotlib.pyplot" in _sys.modules:
    _code_execute_patch_pyplot(_sys.modules["matplotlib.pyplot"])


def set_plot_prefix(prefix: str, work_dir: str) -> None:
  """Bind the plot file prefix and directory for the next cell; restart numbering."""
  _state["prefix"] = prefix
  _state["work_dir"] = str(work_dir)
  _state["counter"] = 0
