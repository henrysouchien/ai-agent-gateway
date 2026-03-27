from pathlib import Path

from agent_gateway import create_agent


NOTES_DIR = Path(__file__).parent / "local_notes"
NOTES_DIR.mkdir(exist_ok=True)


def _safe_path(filename: str) -> Path:
  cleaned = Path(filename).name
  if not cleaned:
    raise ValueError("filename is required")
  return NOTES_DIR / cleaned


async def write_note(tool_input, **_kwargs):
  filename = str(tool_input.get("filename") or "").strip()
  text = str(tool_input.get("text") or "").strip()
  if not filename or not text:
    return None, {"code": "invalid_input", "message": "filename and text are required"}

  path = _safe_path(filename)
  path.write_text(text + "\n", encoding="utf-8")
  return {"ok": True, "path": str(path), "bytes_written": len(text.encode("utf-8"))}, None


async def read_note(tool_input, **_kwargs):
  filename = str(tool_input.get("filename") or "").strip()
  if not filename:
    return None, {"code": "invalid_input", "message": "filename is required"}

  path = _safe_path(filename)
  if not path.exists():
    return None, {"code": "not_found", "message": f"Note not found: {filename}"}
  return {"filename": filename, "text": path.read_text(encoding="utf-8")}, None


app = create_agent(
  "You can store and retrieve short notes with write_note and read_note. "
  "Use the tools whenever the user asks to save or fetch a note.",
  tool_handlers={
    "write_note": write_note,
    "read_note": read_note,
  },
  tool_definitions=[
    {
      "name": "write_note",
      "description": "Write a text note to the local_notes directory.",
      "input_schema": {
        "type": "object",
        "properties": {
          "filename": {"type": "string", "description": "File name such as plan.txt."},
          "text": {"type": "string", "description": "Note content to write."},
        },
        "required": ["filename", "text"],
      },
    },
    {
      "name": "read_note",
      "description": "Read a text note from the local_notes directory.",
      "input_schema": {
        "type": "object",
        "properties": {
          "filename": {"type": "string", "description": "File name such as plan.txt."}
        },
        "required": ["filename"],
      },
    },
  ],
)
