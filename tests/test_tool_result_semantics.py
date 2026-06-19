import agent_gateway.runner as gateway_runner
from agent_gateway.tool_result_semantics import classify_semantic_tool_error, is_semantic_tool_error


def test_classifies_status_error_payload() -> None:
  payload = {
    "status": "error",
    "error": {"code": "not_found", "message": "Ticker X not found"},
  }

  semantic_error = classify_semantic_tool_error(payload)

  assert semantic_error == {
    "code": "tool_status_error",
    "message": "Ticker X not found",
    "source": "status",
    "status": "error",
    "sub_code": "not_found",
  }
  assert is_semantic_tool_error(payload) is True


def test_classifies_success_false_payload() -> None:
  payload = {
    "success": False,
    "reason": "No rows matched",
  }

  semantic_error = classify_semantic_tool_error(payload)

  assert semantic_error == {
    "code": "tool_success_false",
    "message": "No rows matched",
    "source": "success",
    "success": False,
  }


def test_classifies_status_error_payload_with_validation_details() -> None:
  payload = {
    "status": "error",
    "error": "decisions-log entry is invalid",
    "error_code": "invalid_decisions_log_entry",
    "validation_error": True,
    "validation_errors": [
      {"type": "missing", "loc": ["date"], "msg": "Field required"},
      {
        "type": "dict_type",
        "loc": ["patch_ops_applied", 0],
        "msg": "Input should be a valid dictionary",
      },
    ],
    "required_fields": ["date", "skill", "decision", "rationale"],
  }

  semantic_error = classify_semantic_tool_error(payload)

  assert semantic_error is not None
  assert semantic_error["code"] == "tool_status_error"
  assert semantic_error["source"] == "status"
  assert semantic_error["status"] == "error"
  assert semantic_error["sub_code"] == "invalid_decisions_log_entry"
  assert "decisions-log entry is invalid" in semantic_error["message"]
  assert "date: Field required" in semantic_error["message"]
  assert "patch_ops_applied.0: Input should be a valid dictionary" in semantic_error["message"]
  assert "required_fields: date, skill, decision, rationale" in semantic_error["message"]


def test_classifies_status_error_payload_with_errors_alias() -> None:
  payload = {
    "status": "error",
    "error": "price target validation failed",
    "error_type": "price_target_validation_error",
    "errors": [
      {"type": "missing", "loc": ["as_of"], "msg": "Field required"},
      {
        "type": "extra_forbidden",
        "loc": ["ranges", "currency"],
        "msg": "Extra inputs are not permitted",
      },
      {
        "type": "missing",
        "loc": ["driver_sensitivities", 0, "driver_key"],
        "msg": "Field required",
      },
    ],
  }

  semantic_error = classify_semantic_tool_error(payload)

  assert semantic_error is not None
  assert semantic_error["code"] == "tool_status_error"
  assert semantic_error["source"] == "status"
  assert semantic_error["status"] == "error"
  assert semantic_error["sub_code"] == "price_target_validation_error"
  assert "price target validation failed" in semantic_error["message"]
  assert "as_of: Field required" in semantic_error["message"]
  assert "ranges.currency: Extra inputs are not permitted" in semantic_error["message"]
  assert "driver_sensitivities.0.driver_key: Field required" in semantic_error["message"]


def test_warning_only_payload_is_not_semantic_error() -> None:
  assert classify_semantic_tool_error({"status": "success", "warning": "partial data"}) is None
  assert classify_semantic_tool_error({"response": "", "warning": "timed out"}) is None
  assert classify_semantic_tool_error(["status", "error"]) is None


def test_runner_soft_error_wrapper_delegates_to_semantic_helper() -> None:
  error_payload = {"status": "error", "error": "failed"}
  ok_payload = {"status": "success", "warning": "partial data"}

  assert gateway_runner._is_soft_error is is_semantic_tool_error
  assert gateway_runner.AgentRunner._is_soft_error(error_payload) is True
  assert gateway_runner.AgentRunner._is_soft_error(ok_payload) is False


def test_runner_soft_error_wrapper_calls_module_alias(monkeypatch) -> None:
  payload = {"status": "success"}
  calls = []

  def sentinel(result):
    calls.append(result)
    return True

  monkeypatch.setattr(gateway_runner, "_is_soft_error", sentinel)

  assert gateway_runner.AgentRunner._is_soft_error(payload) is True
  assert calls == [payload]
