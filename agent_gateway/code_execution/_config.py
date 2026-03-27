from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class CodeExecutionConfig:
  """Configuration for built-in code execution.

  Fields are grouped into backend selection, script preamble generation,
  environment preparation, working-directory management, output limits, and tool
  description customization.
  """

  # Backend
  docker_image: str = ""
  register_docker: bool = True
  register_subprocess: bool = True

  # Preamble
  preamble_suffix: str = ""
  build_preamble: Callable[[str], str] | None = None

  # Environment
  extra_env: Dict[str, str] | None = None
  prepare_env: Callable[[Dict[str, str]], Dict[str, str]] | None = None

  # Work directory
  work_dir_root: str | None = None
  work_dir_prefix: str = "code_exec_"

  # Limits
  max_output_bytes: int = 100 * 1024
  max_images: int = 5
  max_image_base64_bytes: int = 500 * 1024
  default_timeout_ms: int = 30_000
  max_timeout_ms: int = 120_000

  # Tool schema
  tool_description_suffix: str = ""
