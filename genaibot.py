print("🚀 Starting bot file...")
print("SCRIPT_VERSION = openvino_genai_telegram_fastapi_merged_v1")

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

import openvino_genai as ov_genai

import asyncio
import threading
import traceback
import json
import time
import os
import sys
import re
import html
import ast
import operator
import math
from datetime import datetime


# =========================
# CONFIG
# =========================

TOKEN = "8297145439:AAFGAJhKFF-HaISR8Qu-owSDcB2P8FgkAlQ"

# Your OpenVINO GenAI model folder.
# Example:
# MODEL_PATH = r"C:\models\gemma4"
MODEL_PATH = r"C:\models\gemma4"

# Device options: "GPU", "CPU", "NPU"
DEVICE = "GPU"

# FastAPI server settings
API_HOST = "0.0.0.0"
API_PORT = 9000

# Generation settings
MAX_NEW_TOKENS = 256
PROBE_MAX_NEW_TOKENS = 64
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = 50
DO_SAMPLE = False

# Stability settings
MAX_PROMPT_CHARS = 6000
MAX_HISTORY_MESSAGES = 10

# Agent/tool mode
ENABLE_AGENT_TOOLS = True
MAX_AGENT_STEPS = 3

# Debug logging
PRINT_PROMPTS = True
PRINT_RESPONSES = True


# =========================
# GLOBAL STATE
# =========================

PIPE = None

# One lock protects the single shared model pipeline.
# This prevents Telegram + API requests from hitting GPU at the same time.
MODEL_LOCK = threading.Lock()

CHAT_HISTORY = {}
CHAT_LOCKS = {}

API_APP = FastAPI(title="OpenVINO GenAI Local API")


# =========================
# TEXT CLEANUP
# =========================

def clean_text(text):
    if not text:
        return ""

    text = str(text)
    text = html.unescape(text)

    cleanup_tokens = [
        "<end_of_turn>",
        "<start_of_turn>",
        "<|turn>",
        "<turn|>",
        "<eos>",
        "</s>",
        "<s>",
        "<|image|>",
        "<|audio|>",
        "<image|>",
        "<audio|>",
        "[multimodal]"
    ]

    for token in cleanup_tokens:
        text = text.replace(token, "")

    text = re.sub(r"<unused\d+>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    text = re.sub(r"(?is)<thinking>.*?</thinking>", "", text)
    text = re.sub(r"(?is)<reasoning>.*?</reasoning>", "", text)

    text = text.replace("\x00", "")

    text = "".join(
        ch for ch in text
        if ch == "\n" or ch == "\t" or ord(ch) >= 32
    )

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def sanitize_user_input(text):
    if not text:
        return ""

    text = str(text)
    text = html.unescape(text)
    text = text.replace("\x00", "")

    text = "".join(
        ch for ch in text
        if ch in ["\n", "\t", "\r"] or ord(ch) >= 32
    )

    return text.strip()


# =========================
# CHAT MEMORY
# =========================

def get_lock(chat_id):
    if chat_id not in CHAT_LOCKS:
        CHAT_LOCKS[chat_id] = asyncio.Lock()
    return CHAT_LOCKS[chat_id]


def get_history(chat_id):
    return CHAT_HISTORY.get(chat_id, [])


def add_to_history(chat_id, role, content):
    if chat_id not in CHAT_HISTORY:
        CHAT_HISTORY[chat_id] = []

    cleaned = clean_text(content)

    if not cleaned:
        return

    CHAT_HISTORY[chat_id].append(
        {
            "role": role,
            "content": cleaned
        }
    )

    CHAT_HISTORY[chat_id] = CHAT_HISTORY[chat_id][-MAX_HISTORY_MESSAGES:]


def clear_history(chat_id):
    CHAT_HISTORY[chat_id] = []


def render_memory(chat_id):
    history = get_history(chat_id)

    if not history:
        return "Memory is empty."

    lines = []

    for i, msg in enumerate(history, start=1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "").replace("\n", " ")

        if len(content) > 300:
            content = content[:300] + "..."

        lines.append(f"{i}. {role}: {content}")

    return "\n".join(lines)


def build_chat_prompt(chat_id, user_text):
    history = get_history(chat_id)

    system_prompt = (
        "You are a helpful local AI assistant running through OpenVINO GenAI. "
        "Answer clearly and concisely."
    )

    parts = [f"System: {system_prompt}"]

    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if not content:
            continue

        if role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")

    parts.append(f"User: {user_text}")
    parts.append("Assistant:")

    prompt = "\n".join(parts)

    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[-MAX_PROMPT_CHARS:]

    return prompt


# =========================
# GENERATION CONFIG
# =========================

def safe_set_generation_config(config, name, value):
    try:
        setattr(config, name, value)
        return True
    except Exception:
        return False


def make_generation_config(max_new_tokens=None):
    config = ov_genai.GenerationConfig()

    safe_set_generation_config(
        config,
        "max_new_tokens",
        max_new_tokens if max_new_tokens is not None else MAX_NEW_TOKENS
    )

    safe_set_generation_config(config, "temperature", TEMPERATURE)
    safe_set_generation_config(config, "top_p", TOP_P)
    safe_set_generation_config(config, "top_k", TOP_K)
    safe_set_generation_config(config, "do_sample", DO_SAMPLE)

    return config


def normalize_generation_output(result):
    if result is None:
        return ""

    if isinstance(result, str):
        return result

    if hasattr(result, "texts"):
        try:
            texts = result.texts

            if isinstance(texts, list) and texts:
                return str(texts[0])

            return str(texts)
        except Exception:
            pass

    if hasattr(result, "sequences"):
        try:
            sequences = result.sequences

            if isinstance(sequences, list) and sequences:
                return str(sequences[0])

            return str(sequences)
        except Exception:
            pass

    return str(result)


def generate_raw(prompt, max_new_tokens=None):
    global PIPE

    if PIPE is None:
        return "❌ Model pipeline is not loaded."

    prompt = sanitize_user_input(prompt)

    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[-MAX_PROMPT_CHARS:]

    if PRINT_PROMPTS:
        print("📤 Prompt sent to OpenVINO GenAI:")
        print(prompt[:4000])

    config = make_generation_config(max_new_tokens=max_new_tokens)

    try:
        with MODEL_LOCK:
            result = PIPE.generate(prompt, config)

        text = normalize_generation_output(result)
        text = clean_text(text)

        if PRINT_RESPONSES:
            print("📥 Model response:")
            print(text[:4000])

        if not text:
            return "The model returned an empty response."

        return text

    except Exception as e:
        traceback.print_exc()
        return (
            "❌ OpenVINO GenAI generation error.\n\n"
            f"{type(e).__name__}: {e}"
        )


async def generate_async(prompt, max_new_tokens=None):
    return await asyncio.to_thread(generate_raw, prompt, max_new_tokens)


# =========================
# SIMPLE AGENT TOOLS
# =========================

def tool_time(args):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"Current local datetime: {now}"


SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval_math_node(node):
    if isinstance(node, ast.Expression):
        return safe_eval_math_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)

        if op_type not in SAFE_OPERATORS:
            raise ValueError("Operator not allowed.")

        left = safe_eval_math_node(node.left)
        right = safe_eval_math_node(node.right)

        return SAFE_OPERATORS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)

        if op_type not in SAFE_OPERATORS:
            raise ValueError("Unary operator not allowed.")

        value = safe_eval_math_node(node.operand)

        return SAFE_OPERATORS[op_type](value)

    raise ValueError("Expression not allowed.")


def safe_calculate(expression):
    expression = str(expression).strip()

    if len(expression) > 200:
        raise ValueError("Expression too long.")

    tree = ast.parse(expression, mode="eval")
    value = safe_eval_math_node(tree)

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        raise ValueError("Result is not finite.")

    return value


def tool_calculator(args):
    expression = ""

    if isinstance(args, dict):
        expression = args.get("expression", "")

    if not expression:
        return "Missing expression."

    try:
        result = safe_calculate(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"


def tool_echo(args):
    if isinstance(args, dict):
        return str(args.get("text", ""))
    return str(args)


TOOLS = {
    "time": {
        "description": "Get the current local date and time.",
        "function": tool_time
    },
    "calculator": {
        "description": "Evaluate a basic arithmetic expression. Example args: {\"expression\":\"2+2*5\"}",
        "function": tool_calculator
    },
    "echo": {
        "description": "Echo text back. Example args: {\"text\":\"hello\"}",
        "function": tool_echo
    }
}


def extract_json_object(text):
    if not text:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        return None

    candidate = match.group(0)

    try:
        return json.loads(candidate)
    except Exception:
        return None


def run_tool(tool_name, args):
    if tool_name not in TOOLS:
        return f"Unknown tool: {tool_name}"

    try:
        return TOOLS[tool_name]["function"](args)
    except Exception as e:
        return f"Tool error: {type(e).__name__}: {e}"


def build_agent_instruction(user_text, scratchpad):
    tool_lines = []

    for name, spec in TOOLS.items():
        tool_lines.append(f"- {name}: {spec['description']}")

    tools_text = "\n".join(tool_lines)

    return f"""
You are an agent. You may either answer directly or call one tool.

Available tools:
{tools_text}

You must respond with ONLY valid JSON in one of these two formats:

To call a tool:
{{"tool":"tool_name","args":{{"key":"value"}}}}

To provide the final answer:
{{"final":"your answer here"}}

User request:
{user_text}

Previous tool results:
{scratchpad}
""".strip()


async def run_agent(user_text):
    if not ENABLE_AGENT_TOOLS:
        return await generate_async(user_text)

    scratchpad = ""

    for step in range(1, MAX_AGENT_STEPS + 1):
        agent_prompt = build_agent_instruction(user_text, scratchpad)

        raw = await generate_async(agent_prompt, max_new_tokens=256)
        parsed = extract_json_object(raw)

        if not parsed:
            return (
                "I could not parse the agent response as JSON. "
                "Here is the raw response:\n\n"
                f"{raw}"
            )

        if "final" in parsed:
            return clean_text(parsed.get("final", ""))

        tool_name = parsed.get("tool")
        args = parsed.get("args", {})

        if not tool_name:
            return f"Agent response missing tool name:\n{json.dumps(parsed, indent=2)}"

