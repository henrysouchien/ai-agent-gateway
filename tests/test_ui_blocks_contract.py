from __future__ import annotations

from agent_gateway import ui_blocks_contract


def test_packaged_contract_accessors_are_valid() -> None:
  directory = ui_blocks_contract.packaged_contract_directory()

  assert directory.is_dir()
  assert ui_blocks_contract.contract_version() == 1
  assert ui_blocks_contract.manifest()["contract"]["contract_version"] == 1
  assert ui_blocks_contract.envelope_schema()["$id"] == "hank_ui_blocks.v1"
  assert ui_blocks_contract.fallback_projection_table()["projections"]
  assert ui_blocks_contract.fixtures()
  assert all(
    "expected_code" in fixture
    for fixture in ui_blocks_contract.fixtures()
    if fixture["expectation"] == "reject"
  )
