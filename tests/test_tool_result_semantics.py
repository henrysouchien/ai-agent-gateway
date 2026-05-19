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


def test_warning_only_payload_is_not_semantic_error() -> None:
  assert classify_semantic_tool_error({"status": "success", "warning": "partial data"}) is None
  assert classify_semantic_tool_error({"response": "", "warning": "timed out"}) is None
  assert classify_semantic_tool_error(["status", "error"]) is None
