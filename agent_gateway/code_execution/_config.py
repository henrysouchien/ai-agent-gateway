from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Literal


@dataclass
class CodeExecutionConfig:
  """Configuration for built-in code execution.

  Fields are grouped into kernel placement, script preamble generation,
  environment preparation, working-directory management, output limits, and tool
  description customization.
  """

  # Placement of the session's one Python kernel: "local" runs it on the host,
  # "docker" runs the same kernel in a container. `readonly_mounts` are host
  # directories mounted read-only at the same absolute path in the container.
  placement: Literal["local", "docker"] = "local"
  docker_image: str = ""
  readonly_mounts: tuple[str, ...] = ()

  # Preamble
  preamble_suffix: str = ""

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
  # Recorded beside the measured host wait on a result; never armed as a wait,
  # interrupt or refusal. `default_timeout_ms` bounds compute only.
  default_host_wait_ms: int | None = None
  max_timeout_ms: int = 120_000

  # Tool schema
  tool_description_suffix: str = ""
