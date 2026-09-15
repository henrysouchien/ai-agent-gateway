from __future__ import annotations

from agent_gateway import canvas_kit_contract


def test_packaged_canvas_kit_contract_accessors() -> None:
  assert canvas_kit_contract.packaged_contract_directory().is_dir()
  assert canvas_kit_contract.contract_version() == 1




def test_authoring_manifest_is_generated_from_packaged_types_and_policy() -> None:
  value = canvas_kit_contract.authoring_manifest()

  assert value["generated_from"] == [
    "types/node_modules/@hank/canvas-kit/components/index.d.ts",
    "types/node_modules/@hank/canvas-kit/fmt.d.ts",
    "canvas_build/policy.mjs",
  ]
  assert "title: string" in value["components"]["SectionHeader"]
  assert "children" not in value["components"]["SectionHeader"]
  assert "metrics: MetricStripItem[]" in value["components"]["MetricStrip"]
  assert "items: Array<{ label: string; mark: MarkRole; }>" in value["components"]["MarkLegend"]
  assert "data: Row[]" in value["components"]["DataTable"]
  assert value["components"]["SectionBreak"] == "{} (no props or children)"
  assert "label: string" in value["types"]["MetricStripItem"]
  assert value["formatters"]["fmtPercent"].endswith(": string")
  assert value["module_policy"]["allowed_imports"] == [
    "react", "recharts", "@hank/canvas-kit",
  ]
  assert value["module_policy"]["const_only_module_variables"] is True
  assert value["module_policy"]["literal_module_data_only"] is True


def test_authoring_manifest_prompt_exposes_exact_component_shapes() -> None:
  prompt = canvas_kit_contract.authoring_manifest_prompt()

  assert "Generated read-only Canvas Kit authoring manifest" in prompt
  assert "SectionHeader:" in prompt
  assert "title: string" in prompt
  assert "MetricStripItem" in prompt
  assert "calls, new expressions, and await are forbidden" in prompt


def test_component_repair_hint_names_nearest_component_and_related_item_shape() -> None:
  source = """export default function Example() {
  return <MetricStrip
    items={[]}
  />;
}
"""

  hint = canvas_kit_contract.component_repair_hint(source, 3, "Fix the prop.")

  assert "Expected MetricStrip props:" in hint
  assert "metrics: MetricStripItem[]" in hint
  assert "Related shape: MetricStripItem" in hint
