from __future__ import annotations

from types import SimpleNamespace


def fake_identity_resolver(
  user_id,
  *,
  risk_user_id=None,
  user_email=None,
  role=None,
  channel=None,
  **_kwargs,
):
  numeric_id = int(risk_user_id or (user_id if str(user_id).isdecimal() else 0))
  owner = str(numeric_id) if numeric_id else str(user_id)
  slug = str(user_id) if str(user_id) != owner or not numeric_id else None
  aliases = (owner,)
  if risk_user_id:
    aliases = tuple(dict.fromkeys(value for value in (owner, slug, user_email) if value))
  return SimpleNamespace(
    owner_user_id=owner,
    raw_user_id=str(user_id),
    user_slug=slug,
    risk_user_id=numeric_id,
    user_email=user_email,
    aliases=aliases,
    role=role,
    channel=channel,
    identity_status=(
      "risk_user_id_authoritative" if risk_user_id
      else "numeric_user_id" if numeric_id
      else "legacy_user_id_fallback"
    ),
  )


def fake_mcp_user_key_lookup(user_id, user_email):
  return {
    "key": "test-mcp-key",
    "slug": str(user_id),
    "email": user_email,
    "risk_user_id": int(user_id) if str(user_id).isdecimal() else None,
    "channel": "mcp",
    "role": "owner",
  }
