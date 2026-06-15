from __future__ import annotations

from html.parser import HTMLParser
import json
import re
from pathlib import Path

from schema.html_artifact import HtmlArtifact


_HTML_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9-]{1,128}$")
_FORBIDDEN_TAGS = frozenset({"script", "style", "link", "meta", "base", "iframe", "object", "embed", "form"})
_FORBIDDEN_ATTRS = frozenset({"style", "srcdoc"})


class _HtmlArtifactPolicyParser(HTMLParser):
  def __init__(self) -> None:
    super().__init__(convert_charrefs=True)
    self.errors: list[str] = []

  def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
    self._validate_tag(tag, attrs)

  def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
    self._validate_tag(tag, attrs)

  def _validate_tag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
    normalized_tag = tag.lower()
    if normalized_tag in _FORBIDDEN_TAGS:
      self.errors.append(f"forbidden HTML tag: {normalized_tag}")
    for raw_name, raw_value in attrs:
      name = raw_name.lower()
      if name in _FORBIDDEN_ATTRS or name.startswith("on"):
        self.errors.append(f"forbidden HTML attribute: {name}")
      value = _normalize_url_attribute(raw_value or "")
      if value.startswith("javascript:"):
        self.errors.append(f"forbidden javascript URL in attribute: {name}")


def write_html_artifact(
  *,
  workspace_dir: Path,
  artifact: HtmlArtifact,
  html_content: str,
) -> None:
  """Atomic dual-write: HTML first, sidecar rename as the commit point."""
  artifact_id = _validate_html_artifact_id(artifact.artifact_id)
  validate_html_artifact_content(html_content)
  directory = _html_artifacts_dir(workspace_dir, create=True)

  html_path = _ensure_under_workspace(directory / f"{artifact_id}.html", workspace_dir)
  sidecar_path = _ensure_under_workspace(directory / f"{artifact_id}.json", workspace_dir)
  html_tmp_path = _ensure_under_workspace(directory / f"{artifact_id}.html.tmp", workspace_dir)
  sidecar_tmp_path = _ensure_under_workspace(directory / f"{artifact_id}.json.tmp", workspace_dir)

  payload = json.dumps(
    artifact.model_dump(mode="json"),
    ensure_ascii=False,
    separators=(",", ":"),
  )

  try:
    html_tmp_path.write_text(html_content, encoding="utf-8")
    sidecar_tmp_path.write_text(payload, encoding="utf-8")
    html_tmp_path.replace(html_path)
    sidecar_tmp_path.replace(sidecar_path)
  except Exception:
    _unlink_if_exists(html_tmp_path)
    _unlink_if_exists(sidecar_tmp_path)
    raise


def read_html_artifact_sidecar(workspace_dir: Path, artifact_id: str) -> HtmlArtifact | None:
  sidecar_path = _html_artifact_path(workspace_dir, artifact_id, suffix=".json")
  if not sidecar_path.is_file():
    return None
  return HtmlArtifact.model_validate_json(sidecar_path.read_bytes())


def read_html_artifact_content(workspace_dir: Path, artifact_id: str) -> str | None:
  html_path = _html_artifact_path(workspace_dir, artifact_id, suffix=".html")
  if not html_path.is_file():
    return None
  return html_path.read_text(encoding="utf-8")


def list_html_artifacts(
  workspace_dir: Path,
  *,
  ticker: str | None = None,
  purpose: str | None = None,
  since: float | None = None,
  limit: int = 50,
) -> list[HtmlArtifact]:
  directory = _html_artifacts_dir(workspace_dir)
  if not directory.is_dir() or limit <= 0:
    return []

  artifacts: list[tuple[int, Path, HtmlArtifact]] = []
  for sidecar_path in sorted(directory.glob("*.json"), key=lambda path: path.name):
    try:
      artifact_id = _validate_html_artifact_id(sidecar_path.stem)
      safe_sidecar_path = _ensure_under_workspace(sidecar_path, workspace_dir)
    except ValueError:
      continue
    if artifact_id != sidecar_path.stem:
      continue
    try:
      sidecar_stat = safe_sidecar_path.stat()
      artifact = HtmlArtifact.model_validate_json(safe_sidecar_path.read_bytes())
    except (OSError, ValueError):
      continue
    if ticker is not None and artifact.ticker != ticker:
      continue
    if purpose is not None and artifact.purpose != purpose:
      continue
    if since is not None and sidecar_stat.st_mtime < since:
      continue
    artifacts.append((sidecar_stat.st_mtime_ns, sidecar_path, artifact))

  artifacts.sort(key=lambda entry: (entry[0], entry[1].as_posix()), reverse=True)
  return [artifact for _mtime_ns, _path, artifact in artifacts[:limit]]


def _html_artifact_path(workspace_dir: Path, artifact_id: str, *, suffix: str) -> Path:
  normalized_artifact_id = _validate_html_artifact_id(artifact_id)
  return _ensure_under_workspace(_html_artifacts_dir(workspace_dir) / f"{normalized_artifact_id}{suffix}", workspace_dir)


def validate_html_artifact_content(html_content: str) -> None:
  if not isinstance(html_content, str) or not html_content.strip():
    raise ValueError("html artifact content must be a non-empty string")
  parser = _HtmlArtifactPolicyParser()
  try:
    parser.feed(html_content)
    parser.close()
  except Exception as exc:
    raise ValueError(f"invalid HTML artifact content: {exc}") from exc
  if parser.errors:
    raise ValueError("; ".join(parser.errors))


def _normalize_url_attribute(value: str) -> str:
  without_embedded_controls = value.replace("\t", "").replace("\n", "").replace("\r", "")
  return re.sub(r"^[\x00-\x20]+|[\x00-\x20]+$", "", without_embedded_controls).lower()


def _html_artifacts_dir(workspace_dir: Path, *, create: bool = False) -> Path:
  workspace_root = Path(workspace_dir).resolve()
  artifacts_dir = _ensure_under_workspace(workspace_root / "artifacts", workspace_root)
  if create:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
  elif artifacts_dir.exists():
    _ensure_under_workspace(artifacts_dir, workspace_root)
  directory = _ensure_under_workspace(artifacts_dir / "_html", workspace_root)
  if create:
    directory.mkdir(parents=True, exist_ok=True)
  elif directory.exists():
    _ensure_under_workspace(directory, workspace_root)
  return directory


def _ensure_under_workspace(path: Path, workspace_dir: Path) -> Path:
  workspace_root = Path(workspace_dir).resolve()
  resolved_path = Path(path).resolve()
  try:
    resolved_path.relative_to(workspace_root)
  except ValueError as exc:
    raise ValueError("resolved html artifact path escapes workspace") from exc
  return resolved_path


def _validate_html_artifact_id(artifact_id: str) -> str:
  normalized = str(artifact_id or "").strip()
  if not _HTML_ARTIFACT_ID_RE.match(normalized):
    raise ValueError("invalid html artifact_id")
  return normalized


def _unlink_if_exists(path: Path) -> None:
  try:
    path.unlink()
  except FileNotFoundError:
    pass


__all__ = [
  "list_html_artifacts",
  "read_html_artifact_content",
  "read_html_artifact_sidecar",
  "validate_html_artifact_content",
  "write_html_artifact",
]
