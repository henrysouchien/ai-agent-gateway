"""The signed native-Risk callback payload: one schema, one error mapping.

The producer edge and the gateway runtime both parse the same signed body here, so
an unknown action and a malformed request mean the same thing on either side.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError


class NativeCallbackError(ValueError):
  """A callback body that is not one of the native actions."""

  def __init__(self, code: str, message: str) -> None:
    super().__init__(message)
    self.code = code
    self.message = message


class CatalogCallbackPayload(BaseModel):
  model_config = ConfigDict(frozen=True, extra="forbid")
  action: Literal["catalog"]
  cell_token: str = Field(min_length=1)


class BeginCallbackPayload(BaseModel):
  model_config = ConfigDict(frozen=True, extra="forbid")
  action: Literal["begin"]
  cell_token: str = Field(min_length=1)
  operation: str = Field(min_length=1)
  executed_input: dict[str, Any]
  producer_request_id: str = Field(min_length=1)


class SettleCallbackPayload(BaseModel):
  model_config = ConfigDict(frozen=True, extra="forbid")
  action: Literal["settle"]
  cell_token: str = Field(min_length=1)
  nested_call_id: str = Field(min_length=1)
  producer_request_id: str = Field(min_length=1)
  parent_call_id: str = Field(min_length=1)
  ordinal: int = Field(ge=0)
  result: Any = None
  error: dict[str, Any] | None = None
  provenance: dict[str, Any] | None = None


NativeCallback = CatalogCallbackPayload | BeginCallbackPayload | SettleCallbackPayload

_NATIVE_CALLBACK_ADAPTER: TypeAdapter[NativeCallback] = TypeAdapter(
  Annotated[NativeCallback, Field(discriminator="action")]
)


def parse_native_callback(body: bytes) -> NativeCallback:
  try:
    return _NATIVE_CALLBACK_ADAPTER.validate_json(body)
  except ValidationError as exc:
    for err in exc.errors():
      if err["type"] == "union_tag_invalid":
        raise NativeCallbackError(
          "unknown_native_action",
          f"Unknown native Risk callback action: {(err.get('ctx') or {}).get('tag')!r}.",
        ) from exc
    raise NativeCallbackError(
      "invalid_native_request", "Native Risk callback is malformed.",
    ) from exc
