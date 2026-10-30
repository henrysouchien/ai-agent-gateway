from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_gateway.user_identity_mapping import RunOwnerUnresolved, risk_user_id_for
from agent_gateway.user_runtime_credentials import (
  USER_RUNTIME_CREDENTIALS_FILE_ENV,
  DeskReplica,
  UserRuntimeIdentity,
  mint_user_runtime_credentials,
  write_user_runtime_credentials,
)


def _roster(*entries: dict[str, object]) -> str:
  return json.dumps(list(entries))


def _henry(**overrides: object) -> dict[str, object]:
  entry: dict[str, object] = {
    "key": "mcp-key-for-henry",
    "channel": "mcp",
    "slug": "henry",
    "email": "henry@example.com",
    "risk_user_id": 1,
    "role": "owner",
  }
  entry.update(overrides)
  return entry


def _bundle_environ(
  tmp_path: Path,
  *,
  slug: str = "henry",
  email: str = "henry@example.com",
  risk_user_id: int = 1,
  storage_id: str = "desk-1",
) -> dict[str, str]:
  path = tmp_path / "bundle.json"
  bundle = mint_user_runtime_credentials(
    org_claim_key="org-secret",
    storage_id=storage_id,
    generation=1,
    identity=UserRuntimeIdentity("henry-key", "excel", slug, email, risk_user_id, "owner", ()),
    desk=DeskReplica("file:///replica", None, None, None, None),
  )
  write_user_runtime_credentials(bundle, path)
  return {USER_RUNTIME_CREDENTIALS_FILE_ENV: str(path)}


def test_numeric_storage_id_maps_to_itself() -> None:
  assert risk_user_id_for("42", environ={}) == 42
  assert risk_user_id_for(" 7 ", environ={}) == 7
  assert risk_user_id_for("042", environ={}) == 42
  # A numeric storage id is itself even when the roster maps that token as a slug.
  disagreeing = {"GATEWAY_USER_KEYS": _roster(_henry(slug="42", risk_user_id=9))}
  assert risk_user_id_for("42", environ=disagreeing) == 42


def test_slug_and_email_via_bundle(tmp_path: Path) -> None:
  environ = _bundle_environ(tmp_path, risk_user_id=9, storage_id="desk-9")
  assert risk_user_id_for("henry", environ=environ) == 9
  assert risk_user_id_for("Henry@Example.com", environ=environ) == 9
  # storage_id is not a risk_user_id unless it is a positive decimal token.
  with pytest.raises(RunOwnerUnresolved):
    risk_user_id_for("desk-9", environ=environ)


def test_bundle_shadows_roster(tmp_path: Path) -> None:
  environ = _bundle_environ(tmp_path, slug="ada", email="ada@example.com", risk_user_id=9)
  environ["GATEWAY_USER_KEYS"] = _roster(_henry(slug="ada", email="ada@example.com", risk_user_id=2))
  assert risk_user_id_for("ada", environ=environ) == 9
  with pytest.raises(RunOwnerUnresolved):
    risk_user_id_for("henry", environ=environ)


def test_slug_and_email_via_roster() -> None:
  environ = {
    "GATEWAY_USER_KEYS": _roster(_henry(channel="web", role="invite")),
  }
  assert risk_user_id_for("henry", environ=environ) == 1
  assert risk_user_id_for(" henry ", environ=environ) == 1
  assert risk_user_id_for("HENRY@example.com", environ=environ) == 1


def test_environ_defaults_to_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delenv(USER_RUNTIME_CREDENTIALS_FILE_ENV, raising=False)
  monkeypatch.setenv("GATEWAY_USER_KEYS", _roster(_henry()))
  assert risk_user_id_for("henry") == 1


def test_unknown_identifier_raises() -> None:
  for identifier, environ in (
    ("nobody", {}),
    ("", {"GATEWAY_USER_KEYS": _roster(_henry())}),
    ("local-user", {"GATEWAY_USER_KEYS": _roster(_henry())}),
  ):
    with pytest.raises(RunOwnerUnresolved) as excinfo:
      risk_user_id_for(identifier, environ=environ)
    assert excinfo.value.reason == "unknown"


def test_roster_entry_without_risk_user_id_raises() -> None:
  entry = _henry()
  del entry["risk_user_id"]
  with pytest.raises(RunOwnerUnresolved) as excinfo:
    risk_user_id_for("henry", environ={"GATEWAY_USER_KEYS": _roster(entry)})
  assert excinfo.value.reason == "unknown"


@pytest.mark.parametrize("identifier", ["0", "00", "-1", "-0", "  -2 ", " 0 "])
def test_non_positive_identifier_raises(identifier: str) -> None:
  roster = {"GATEWAY_USER_KEYS": _roster(_henry(slug="0", email="0@example.com", risk_user_id=4))}
  with pytest.raises(RunOwnerUnresolved) as excinfo:
    risk_user_id_for(identifier, environ=roster)
  assert excinfo.value.reason == "non_positive"


def test_ambiguous_slug_refuses() -> None:
  environ = {
    "GATEWAY_USER_KEYS": _roster(
      _henry(),
      _henry(key="other", email="other@example.com", risk_user_id=2, channel="web"),
    ),
  }
  with pytest.raises(RunOwnerUnresolved) as excinfo:
    risk_user_id_for("henry", environ=environ)
  assert excinfo.value.reason == "ambiguous"
  assert isinstance(excinfo.value.__cause__, ValueError)


@pytest.mark.parametrize("raw", ["{not-json", '{"not": "array"}'])
def test_malformed_roster_refuses_visibly(raw: str) -> None:
  with pytest.raises(RunOwnerUnresolved) as excinfo:
    risk_user_id_for("henry", environ={"GATEWAY_USER_KEYS": raw})
  assert excinfo.value.reason == "malformed_roster"
  assert isinstance(excinfo.value.__cause__, ValueError)
