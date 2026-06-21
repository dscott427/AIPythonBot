print("🚀 Starting bot file...")
print("SCRIPT_VERSION = gemma_ovms_curl_backend_v1")

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import asyncio
import json
import traceback
import subprocess
import re


# =========================
# CONFIG
# =========================

TOKEN = "................................"

AI_URL = "http://localhost:9000/v3/chat/completions"
MODEL_NAME = "gemma"

MAX_COMPLETION_TOKENS = 128
TEMPERATURE = 0.0
REQUEST_TIMEOUT_SECONDS = 180

PRINT_PAYLOADS = True
PRINT_RAW_RESPONSES = True


# =========================
# MEMORY - LOCAL ONLY
# =========================

CHAT_HISTORY = {}
CHAT_LOCKS = {}


def get_lock(chat_id):
    if chat_id not in CHAT_LOCKS:
        CHAT_LOCKS[chat_id] = asyncio.Lock()
    return CHAT_LOCKS[chat_id]


def get_history(chat_id):
    return CHAT_HISTORY.get(chat_id, [])


def clean_text_for_storage(text):
    if not text:
        return ""

    text = str(text)

    cleanup_tokens = [
        "<end_of_turn>",
        "<start_of_turn>",
        "<|turn>",
        "<turn|>",
        "<eos>",
        "</s>",
        "<|image|>",
        "<|audio|>",
        "<image|>",
        "<audio|>",
        "[multimodal]"
    ]

    for token in cleanup_tokens:
        text = text.replace(token, "")

    text = re.sub(r"<unused\d+>", "", text)
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    text = re.sub(r"(?is)<thinking>.*?</thinking>", "", text)
    text = re.sub(r"(?is)<reasoning>.*?</reasoning>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def add_to_history(chat_id, role, content):
    if chat_id not in CHAT_HISTORY:
        CHAT_HISTORY[chat_id] = []

    CHAT_HISTORY[chat_id].append(
        {
            "role": role,
            "content": clean_text_for_storage(content)
        }
    )

    CHAT_HISTORY[chat_id] = CHAT_HISTORY[chat_id][-10:]


def clear_history(chat_id):
    CHAT_HISTORY[chat_id] = []


def render_memory(chat_id):
    history = get_history(chat_id)

    if not history:
        return "Memory is empty."

    lines = []

    for i, msg in enumerate(history, start=1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        content = content.replace("\n", " ")

        if len(content) > 300:
            content = content[:300] + "..."

        lines.append(f"{i}. {role}: {content}")

    return "\n".join(lines)


# =========================
# RESPONSE CLEANUP
# =========================

def clean_reply(text):
    cleaned = clean_text_for_storage(text)

    if not cleaned:
        return "The model returned an empty response."

    return cleaned


# =========================
# PAYLOAD
# This matches your working curl request.
# =========================

def build_payload(user_text):
    return {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_text
                    }
                ]
            }
        ],
        "temperature": TEMPERATURE,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "stream": False
    }


# =========================
# CURL-BASED OVMS CALL
# =========================

def call_ai_with_curl(user_text):
    print("🧠 Calling AI through curl.exe...")

    payload = build_payload(user_text)

    payload_json = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":")
    )

    if PRINT_PAYLOADS:
        print("📤 Payload sent to OVMS:")
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    curl_command = [
        "curl.exe",
        AI_URL,
        "-H",
        "Content-Type: application/json",
        "-d",
        payload_json
    ]

    print("📤 curl command equivalent:")
    print(" ".join(curl_command))

    try:
        result = subprocess.run(
            curl_command,
            capture_output=True,
            text=True,
            timeout=REQUEST_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        return "❌ curl.exe timed out while talking to OVMS."
    except FileNotFoundError:
        return "❌ curl.exe was not found. Windows should include curl.exe, but it is not available in PATH."
    except Exception as e:
        return f"❌ Error running curl.exe:\n{e}"

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    print(f"📥 curl return code: {result.returncode}")

    if stderr:
        print("📥 curl stderr:")
        print(stderr[:4000])

    if PRINT_RAW_RESPONSES:
        print("📥 curl stdout / raw OVMS response:")
        print(stdout[:4000])

    if result.returncode != 0:
        return (
            "❌ curl.exe failed.\n\n"
            f"Return code: {result.returncode}\n\n"
            f"stderr:\n{stderr}\n\n"
            f"stdout:\n{stdout}"
        )

    if not stdout:
        return "❌ OVMS returned an empty response."

    try:
        data = json.loads(stdout)
    except Exception:
        return f"❌ OVMS returned non-JSON response:\n{stdout}"

    if "error" in data:
        return (
            "❌ AI Error from OVMS:\n"
            f"{data}\n\n"
            "Payload that caused it:\n"
            f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
        )

    try:
        reply = data["choices"][0]["message"]["content"]
    except Exception:
        return f"❌ Bad response format:\n{data}"

    return clean_reply(reply)


# =========================
# TELEGRAM HELPERS
# =========================

async def send_long(update, text):
    if not text:
        await update.message.reply_text("Empty response.")
        return

    max_len = 3900

    for i in range(0, len(text), max_len):
        await update.message.reply_text(text[i:i + max_len])


# =========================
# COMMANDS
# =========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ Gemma bot ready.\n\n"
        "This version calls OVMS using curl.exe internally, matching your working curl test.\n\n"
        "Commands:\n"
        "/probe - send the exact known-good test prompt\n"
        "/debug - show backend config\n"
        "/reset - clear local memory\n"
        "/memory - show local memory"
    )


async def probe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    lock = get_lock(chat_id)

    async with lock:
        await update.message.chat.send_action("typing")

        test_text = "Say hello in one sentence."

        try:
            reply = await asyncio.to_thread(call_ai_with_curl, test_text)
        except Exception as e:
            traceback.print_exc()
            reply = f"❌ Probe crash:\n{e}"

        await send_long(update, reply)


async def debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    info = {
        "SCRIPT_VERSION": "gemma_ovms_curl_backend_v1",
        "AI_URL": AI_URL,
        "MODEL_NAME": MODEL_NAME,
        "MAX_COMPLETION_TOKENS": MAX_COMPLETION_TOKENS,
        "TEMPERATURE": TEMPERATURE,
        "backend": "curl.exe",
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
            ],
            "temperature": TEMPERATURE,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "stream": False
        }
    }

    await update.message.reply_text(json.dumps(info, indent=2))


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    clear_history(chat_id)
    await update.message.reply_text("✅ Memory cleared.")


async def memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await send_long(update, render_memory(chat_id))


# =========================
# NORMAL MESSAGE HANDLER
# =========================

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("📨 Message received")

    chat_id = update.effective_chat.id
    user_text = update.message.text

    if not user_text:
        await update.message.reply_text("Please send text.")
        return

    lock = get_lock(chat_id)

    async with lock:
        await update.message.chat.send_action("typing")

        try:
            reply = await asyncio.to_thread(call_ai_with_curl, user_text)
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
    print("SCRIPT_VERSION = gemma_ovms_curl_backend_v1")

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
    main()
