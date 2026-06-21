print("🚀 Starting bot file...")
print("SCRIPT_VERSION = gemma_ovms_curl_stdin_payload_v12")

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

import asyncio
import json
import traceback
import re
import html
import subprocess


# =========================
# CONFIG
# =========================

TOKEN = "PASTE_YOUR_TELEGRAM_BOT_TOKEN_HERE"

# Use explicit IPv4 loopback instead of localhost.
AI_URL = "http://127.0.0.1:9000/v3/chat/completions"

MODEL_NAME = "gemma"

REQUEST_TIMEOUT_SECONDS = 180
MAX_AI_ATTEMPTS = 2

MAX_INPUT_CHARS = 2500
MAX_HISTORY_MESSAGES = 10

PRINT_PAYLOADS = True
PRINT_RAW_RESPONSES = True


# =========================
# LOCAL MEMORY ONLY
# =========================

CHAT_HISTORY = {}
CHAT_LOCKS = {}


def get_lock(chat_id):
    if chat_id not in CHAT_LOCKS:
        CHAT_LOCKS[chat_id] = asyncio.Lock()
    return CHAT_LOCKS[chat_id]


def get_history(chat_id):
    return CHAT_HISTORY.get(chat_id, [])


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

    text = text.strip()

    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS] + "\n\n[Input truncated for OVMS safety.]"

    return text


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


# =========================
# OVMS PAYLOAD
# This matches your working payload shape.
# =========================

def content_array(text):
    return [
        {
            "type": "text",
            "text": text
        }
    ]


def build_payload(user_text):
    final_prompt = sanitize_user_input(user_text)

    return {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": content_array(final_prompt)
            }
        ]
    }


# =========================
# ERROR DETECTION
# =========================

def is_mediapipe_failure_text(text):
    if not text:
        return False

    markers = [
        "Mediapipe execution failed",
        "LLMExecutor",
        "INVALID_ARGUMENT",
        "Request processing failed",
        "CalculatorGraph::Run() failed"
    ]

    return any(marker in text for marker in markers)


# =========================
# OVMS CURL CALL THROUGH STDIN
# Key fix:
#   curl.exe --data-binary @-
# and payload JSON is sent through stdin as UTF-8 bytes.
# =========================

def call_ai_once(user_text, attempt_number):
    print(f"🧠 Calling OVMS through curl.exe stdin attempt {attempt_number}/{MAX_AI_ATTEMPTS}")
    print(f"🔗 AI_URL = {AI_URL}")
    print(f"🤖 MODEL_NAME = {MODEL_NAME}")

    payload = build_payload(user_text)

    payload_json = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":")
    )

    if PRINT_PAYLOADS:
        print("📤 Payload sent to OVMS:")
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"📏 Payload JSON length: {len(payload_json)}")

    curl_command = [
        "curl.exe",
        "--silent",
        "--show-error",
        "--no-buffer",
        "--http1.1",
        AI_URL,
        "-H",
        "Content-Type: application/json",
        "--data-binary",
        "@-"
    ]

    print("📤 curl command equivalent:")
    print(" ".join(curl_command))
    print("📤 JSON is being piped to curl.exe over stdin, not passed as a command-line argument.")

    try:
        process = subprocess.Popen(
            curl_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        stdout_bytes, stderr_bytes = process.communicate(
            input=payload_json.encode("utf-8"),
            timeout=REQUEST_TIMEOUT_SECONDS
        )

        return_code = process.returncode

    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except Exception:
            pass

        return {
            "ok": False,
            "status": None,
            "body": "curl timeout",
            "message": "❌ curl.exe timed out while talking to OVMS."
        }

    except FileNotFoundError:
        return {
            "ok": False,
            "status": None,
            "body": "curl.exe not found",
            "message": (
                "❌ curl.exe was not found.\n\n"
                "Try running this in CMD:\n"
                "curl.exe --version"
            )
        }

    except Exception as e:
        traceback.print_exc()

        return {
            "ok": False,
            "status": None,
            "body": str(e),
            "message": f"❌ Error running curl.exe:\n{e}"
        }

    stdout = stdout_bytes.decode("utf-8", errors="replace").strip() if stdout_bytes else ""
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip() if stderr_bytes else ""

    print(f"📥 curl return code: {return_code}")
    print(f"📏 curl stdout length: {len(stdout)}")

    if stderr:
        print("📥 curl stderr:")
        print(stderr[:4000])

    if stdout:
        print("📥 Last 500 chars of curl stdout:")
        print(stdout[-500:])

    if PRINT_RAW_RESPONSES:
        print("📥 Raw curl stdout:")
        print(stdout[:8000])

    if return_code != 0:
        return {
            "ok": False,
            "status": return_code,
            "body": stdout or stderr,
            "message": (
                "❌ curl.exe failed.\n\n"
                f"Return code: {return_code}\n\n"
                f"stderr:\n{stderr}\n\n"
                f"stdout:\n{stdout}"
            )
        }

    if not stdout:
        return {
            "ok": False,
            "status": return_code,
            "body": "",
            "message": "❌ OVMS returned an empty response."
        }

    try:
        data = json.loads(stdout)
    except Exception:
        return {
            "ok": False,
            "status": return_code,
            "body": stdout,
            "message": (
                "❌ OVMS returned non-JSON response.\n\n"
                f"Raw response length: {len(stdout)}\n\n"
                f"Raw response:\n{stdout}"
            )
        }

    if "error" in data:
        error_text = json.dumps(data, indent=2, ensure_ascii=False)

        return {
            "ok": False,
            "status": return_code,
            "body": error_text,
            "message": (
                "❌ AI error from OVMS:\n"
                f"{error_text}"
            )
        }

    try:
        reply = data["choices"][0]["message"]["content"]
    except Exception:
        formatted = json.dumps(data, indent=2, ensure_ascii=False)

        return {
            "ok": False,
            "status": return_code,
            "body": formatted,
            "message": (
                "❌ Bad response format from OVMS.\n\n"
                f"{formatted}"
            )
        }

    print(f"📏 Extracted assistant reply length: {len(str(reply))}")

    cleaned = clean_text(reply)

    if not cleaned:
        cleaned = "The model returned an empty response."

    return {
        "ok": True,
        "status": return_code,
        "body": stdout,
        "message": cleaned
    }


def call_ai(user_text):
    last_result = None

    for attempt in range(1, MAX_AI_ATTEMPTS + 1):
        result = call_ai_once(user_text, attempt)
        last_result = result

        if result.get("ok"):
            return result["message"]

        body = result.get("body", "")
        message = result.get("message", "")

        if is_mediapipe_failure_text(body) or is_mediapipe_failure_text(message):
            print("⚠️ OVMS MediaPipe / LLMExecutor failure detected.")

            if attempt < MAX_AI_ATTEMPTS:
                print("🔁 Retrying same stdin curl payload...")
                continue

        break

    if not last_result:
        return "❌ Unknown OVMS error."

    body = last_result.get("body", "")
    message = last_result.get("message", "")

    if is_mediapipe_failure_text(body) or is_mediapipe_failure_text(message):
        return (
            "❌ OVMS MediaPipe / LLMExecutor error.\n\n"
            "The bot used curl.exe and piped the JSON through stdin using --data-binary @-, "
            "but OVMS still rejected the request.\n\n"
            "At this point, the remaining difference is likely OVMS runtime state or request timing, "
            "not JSON escaping.\n\n"
            "Most recent OVMS error:\n"
            f"{body}"
        )

    return last_result.get("message", "❌ Unknown OVMS error.")


# =========================
# TELEGRAM HELPERS
# =========================

def split_text_safely(text, max_len=3500):
    if not text:
        return [""]

    text = str(text)

    if len(text) <= max_len:
        return [text]

    parts = []
    current = ""

    paragraphs = text.split("\n")

    for paragraph in paragraphs:
        if len(paragraph) > max_len:
            if current:
                parts.append(current)
                current = ""

            for i in range(0, len(paragraph), max_len):
                parts.append(paragraph[i:i + max_len])

            continue

        candidate = paragraph if not current else current + "\n" + paragraph

        if len(candidate) > max_len:
            if current:
                parts.append(current)
            current = paragraph
        else:
            current = candidate

    if current:
        parts.append(current)

    return parts


async def send_long(update, text):
    if not update.message:
        return

    if not text:
        await update.message.reply_text("Empty response.", parse_mode=None)
        return

    text = str(text)

    print(f"📏 Telegram outgoing text length: {len(text)}")
    print("📤 Last 300 chars being sent to Telegram:")
    print(text[-300:])

    parts = split_text_safely(text, max_len=3500)

    for i, part in enumerate(parts, start=1):
        if len(parts) > 1:
            message = f"(Part {i}/{len(parts)})\n{part}"
        else:
            message = part

        await update.message.reply_text(
            message,
            parse_mode=None,
            disable_web_page_preview=True
        )

        if len(parts) > 1:
            await asyncio.sleep(0.25)


# =========================
# TELEGRAM COMMANDS
# =========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ Gemma bot ready.\n\n"
        "This version uses curl.exe with JSON piped over stdin using --data-binary @-.\n\n"
        "Commands:\n"
        "/probe - send a known-good test prompt\n"
        "/debug - show bot config\n"
        "/reset - clear local memory\n"
        "/memory - show local memory",
        parse_mode=None
    )


async def probe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    lock = get_lock(chat_id)

    async with lock:
        await update.message.chat.send_action("typing")

        test_text = "Say hello."

        try:
            reply = await asyncio.to_thread(call_ai, test_text)
        except Exception as e:
            traceback.print_exc()
            reply = f"❌ Probe crash:\n{e}"

        await send_long(update, reply)


async def debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    info = {
        "SCRIPT_VERSION": "gemma_ovms_curl_stdin_payload_v12",
        "AI_URL": AI_URL,
        "MODEL_NAME": MODEL_NAME,
        "backend": "curl.exe via stdin using --data-binary @-",
        "REQUEST_TIMEOUT_SECONDS": REQUEST_TIMEOUT_SECONDS,
        "MAX_INPUT_CHARS": MAX_INPUT_CHARS,
        "MAX_AI_ATTEMPTS": MAX_AI_ATTEMPTS,
        "MAX_HISTORY_MESSAGES": MAX_HISTORY_MESSAGES,
        "payload_shape": {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "example"
                        }
                    ]
                }
            ]
        }
    }

    await update.message.reply_text(
        json.dumps(info, indent=2),
        parse_mode=None
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    clear_history(chat_id)
    await update.message.reply_text("✅ Memory cleared.", parse_mode=None)


async def memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await send_long(update, render_memory(chat_id))


# =========================
# NORMAL MESSAGE HANDLER
# =========================

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("📨 Message received")

    if not update.message:
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text

    if not user_text:
        await update.message.reply_text("Please send text.", parse_mode=None)
        return

    lock = get_lock(chat_id)

    async with lock:
        await update.message.chat.send_action("typing")

        try:
            reply = await asyncio.to_thread(call_ai, user_text)
        except Exception as e:
            print("🔥 Exception in handle():")
            traceback.print_exc()
            reply = f"❌ Bot crash while calling AI:\n{e}"

        add_to_history(chat_id, "user", user_text)
        add_to_history(chat_id, "assistant", reply)

        await send_long(update, reply)


# =========================
# MAIN
# =========================

def main():
    print("⚙️ Entering main()")
    print("SCRIPT_VERSION = gemma_ovms_curl_stdin_payload_v12")
    print(f"AI_URL = {AI_URL}")
    print(f"MODEL_NAME = {MODEL_NAME}")

    if TOKEN == "PASTE_YOUR_TELEGRAM_BOT_TOKEN_HERE":
        print("❌ Telegram TOKEN is still the placeholder.")
        print("❌ Edit bot.py and paste your actual Telegram bot token.")
        input("Press Enter to exit...")
        return

    try:
        print("🔧 Building Telegram app...")
        app = Application.builder().token(TOKEN).build()

        print("✅ Adding handlers...")
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("probe", probe))
        app.add_handler(CommandHandler("debug", debug))
        app.add_handler(CommandHandler("reset", reset))
        app.add_handler(CommandHandler("memory", memory))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

        print("🚀 Starting polling...")
        app.run_polling()

    except Exception:
        print("🔥 CRASH IN MAIN:")
        traceback.print_exc()
        input("Press Enter to exit...")


if __name__ == "__main__":
    print("✅ __main__ guard reached. Calling main()...")
    main()