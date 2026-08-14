from __future__ import annotations

from agent_gateway import canvas_kit_contract
from agent_gateway.control_plane import canvas_artifacts


def test_packaged_canvas_kit_contract_accessors() -> None:
  assert canvas_kit_contract.packaged_contract_directory().is_dir()
  assert canvas_kit_contract.contract_version() == 1
  assert canvas_kit_contract.externals_map() == {
    "react": "HankCanvasRuntime.React",
    "recharts": "HankCanvasRuntime.Recharts",
    "@hank/canvas-kit": "HankCanvasRuntime.Kit",
  }


def test_route_caps_are_pinned_to_shared_manifest_limits() -> None:
  limits = canvas_kit_contract.limits()
  assert canvas_artifacts.CANVAS_RENDER_FAILURE_REQUEST_MAX_BYTES == limits["render_failure_post_body_max_bytes"]
  assert canvas_artifacts.CANVAS_RENDER_FAILURE_STORED_CAP == limits["render_failure_stored_max_per_artifact"]
  assert canvas_artifacts.CANVAS_RENDER_FAILURE_REPORTS_PER_RENDER == limits["render_error_reports_max_per_render"]
  assert canvas_artifacts.CANVAS_RENDER_FAILURE_MESSAGE_MAX_CHARS == limits["render_error_message_max_chars"]
  assert canvas_artifacts.CANVAS_RENDER_FAILURE_COMPONENT_STACK_MAX_CHARS == limits["render_error_component_stack_max_chars"]


def test_bundle_format_pins_classic_iife_compiler_contract() -> None:
  assert canvas_kit_contract.bundle_format()["compiler"] == {
    "jsx": "transform", "target": "es2020", "charset": "ascii",
    "legalComments": "none", "minify": True, "format": "iife",
  }


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
