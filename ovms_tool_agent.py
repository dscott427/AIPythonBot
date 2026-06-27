import json
import re
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

OVMS_BASE_URL = "http://localhost:9200/v3"
OVMS_MODEL = "qwen3"

WORKSPACE = Path.cwd().resolve()

DEBUG = True

client = OpenAI(
    base_url=OVMS_BASE_URL,
    api_key="unused",
)


# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT = """
You are a local coding agent.

You have access to tools, but you do not have native tool-calling support.
Instead, you must use this JSON protocol.

You must output exactly one JSON object and nothing else.

If you need to use a tool, respond ONLY with valid JSON in this exact shape:

{
  "tool": "tool_name",
  "arguments": {
    "key": "value"
  }
}

If the task is complete, respond ONLY with valid JSON in this exact shape:

{
  "final": "your final response"
}

Available tools:

1. list_dir
Arguments:
{
  "path": "relative path inside the workspace"
}

2. read_file
Arguments:
{
  "path": "relative path inside the workspace"
}

3. write_file
Arguments:
{
  "path": "relative path inside the workspace",
  "content": "complete file content"
}

Rules:
- Output exactly one JSON object.
- Do not output Markdown.
- Do not output HTML.
- Do not output <br>.
- Do not output <think> tags.
- Do not include explanations outside JSON.
- Never write outside the workspace.
- For creating files, use write_file.
- For editing files, first use read_file, then use write_file with the complete updated file content.
- If a tool returns an error, decide what to do next using another JSON response.
"""


# ============================================================
# Logging
# ============================================================

def debug_print(title: str, value: Any = None) -> None:
    if not DEBUG:
        return

    print(f"\n--- {title} ---", flush=True)

    if value is not None:
        print(value, flush=True)


# ============================================================
# Safety helpers
# ============================================================

def safe_path(path: str) -> Path:
    """
    Resolve a model-provided path safely inside WORKSPACE.
    Blocks attempts to read/write outside the project folder.
    """
    if not path:
        path = "."

    target = (WORKSPACE / path).resolve()

    try:
        target.relative_to(WORKSPACE)
    except ValueError:
        raise ValueError(f"Blocked path outside workspace: {path}")

    return target


# ============================================================
# Tools
# ============================================================

def list_dir(arguments: dict[str, Any]) -> str:
    path = safe_path(str(arguments.get("path", ".")))

    if not path.exists():
        return f"Directory does not exist: {path.relative_to(WORKSPACE)}"

    if not path.is_dir():
        return f"Not a directory: {path.relative_to(WORKSPACE)}"

    entries: list[str] = []

    for item in sorted(path.iterdir(), key=lambda p: p.name.lower()):
        suffix = "/" if item.is_dir() else ""
        entries.append(f"{item.name}{suffix}")

    return "\n".join(entries)


def read_file(arguments: dict[str, Any]) -> str:
    path = safe_path(str(arguments["path"]))

    if not path.exists():
        return f"File does not exist: {path.relative_to(WORKSPACE)}"

    if not path.is_file():
        return f"Not a file: {path.relative_to(WORKSPACE)}"

    return path.read_text(encoding="utf-8")


def write_file(arguments: dict[str, Any]) -> str:
    path = safe_path(str(arguments["path"]))
    content = str(arguments["content"])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    return f"Wrote file: {path.relative_to(WORKSPACE)}"


TOOLS = {
    "list_dir": list_dir,
    "read_file": read_file,
    "write_file": write_file,
}


# ============================================================
# OVMS call
# ============================================================

def call_ovms(messages: list[dict[str, str]]) -> str:
    """
    Call OVMS through the OpenAI-compatible chat/completions endpoint.

    Intended for Qwen3 served by OVMS with:
      --task text_generation
    """
    debug_print("SENDING TO OVMS", json.dumps(messages, indent=2, ensure_ascii=False))

    response = client.chat.completions.create(
        model=OVMS_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
        stream=False,
    )

    debug_print("RAW OVMS RESPONSE OBJECT", response)

    content = response.choices[0].message.content

    if content is None:
        return ""

    return content.strip()


# ============================================================
# JSON cleanup helpers
# ============================================================

def remove_think_blocks(raw: str) -> str:
    """
    Qwen models may emit:
      <think> ... </think>

    Strip those before JSON parsing.
    """
    return re.sub(
        r"<think>.*?</think>",
        "",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )


def remove_markdown_fences(raw: str) -> str:
    raw = raw.strip()

    # Remove opening fences
    raw = re.sub(r"^```(?:json|JSON)?", "", raw).strip()

    # Remove closing fence
    raw = re.sub(r"```$", "", raw).strip()

    return raw


def remove_html_artifacts(raw: str) -> str:
    replacements = {
        "<br>": "",
        "<br/>": "",
        "<br />": "",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#34;": '"',
        "&#39;": "'",
        "&amp;": "&",
    }

    for old, new in replacements.items():
        raw = raw.replace(old, new)

    return raw.strip()

def extract_json(raw: str):
    raw = raw.strip()

    # remove think blocks
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)

    # remove HTML artifacts
    raw = raw.replace("<br>", "")
    raw = raw.replace("&lt;", "<")
    raw = raw.replace("&gt;", ">")

    raw = raw.strip()

    try:
        return json.loads(raw)
    except:
        # fallback: extract JSON object
        start = raw.find("{")
        end = raw.rfind("}")

        if start != -1 and end != -1:
            return json.loads(raw[start:end+1])

    raise Exception("Could not parse JSON:\n" + raw)

def run_agent(user_prompt: str, max_steps: int = 10) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for step in range(max_steps):
        raw = call_ovms(messages)

        print(f"\n--- MODEL STEP {step + 1} ---", flush=True)
        print(raw, flush=True)

        try:
            action = extract_json(raw)
        except Exception as e:
            return f"JSON parse error:\n{e}\n\nraw:\n{raw}"

        if "final" in action:
            return action["final"]

        tool_name = action.get("tool")
        arguments = action.get("arguments", {})

        tool = TOOLS.get(tool_name)

        if tool:
            result = tool(arguments)
        else:
            result = f"Unknown tool: {tool_name}"

        print("\n--- TOOL RESULT ---", flush=True)
        print(result, flush=True)

        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": json.dumps({
                "tool_result": result
            })
        })

    return "Max steps reached"

def main():
    print("SCRIPT STARTED", flush=True)

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = input("Task: ").strip()

    if not prompt:
        print("No task provided.", flush=True)
        return

    print(f"Workspace: {WORKSPACE}", flush=True)

    final = run_agent(prompt)

    print("\n--- FINAL ---", flush=True)
    print(final, flush=True)


if __name__ == "__main__":
    main()