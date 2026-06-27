print("🚀 Starting bot file...")
print("SCRIPT_VERSION = genai_bot_minimal_v2")


# =========================
# IMPORTS
# =========================

import asyncio
import threading
import traceback
import os
import sys

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from fastapi import FastAPI, Request
import uvicorn

import openvino_genai as ov_genai


# =========================
# CONFIG
# =========================

TOKEN = "8297145439:AAFGAJhKFF-HaISR8Qu-owSDcB2P8FgkAlQ"
MODEL_PATH = r"C:\models\gemma4"
DEVICE = "GPU"

API_PORT = 9000


# =========================
# GLOBALS
# =========================

PIPE = None
MODEL_LOCK = threading.Lock()
APP_FASTAPI = FastAPI()


# =========================
# MODEL
# =========================

def load_model():
    global PIPE

    print("🧠 Loading model...")
    print("MODEL_PATH =", MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        print("❌ MODEL_PATH NOT FOUND")
        input("Press Enter to exit...")
        sys.exit(1)

    PIPE = ov_genai.LLMPipeline(MODEL_PATH, DEVICE)

    print("✅ Model loaded.")


def generate(prompt):
    with MODEL_LOCK:
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 200
        result = PIPE.generate(prompt, config)

    return str(result)


async def generate_async(prompt):
    return await asyncio.to_thread(generate, prompt)


# =========================
# FASTAPI
# =========================

@APP_FASTAPI.get("/")
def root():
    return {"status": "OK"}


@APP_FASTAPI.get("/health")
def health():
    return {"status": "OK", "model": PIPE is not None}


@APP_FASTAPI.post("/generate")
async def api_generate(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "")

    if not prompt:
        return {"error": "missing prompt"}

    result = await generate_async(prompt)
    return {"response": result}


def start_api():
    print(f"🌐 API running on port {API_PORT}")
    uvicorn.run(APP_FASTAPI, host="0.0.0.0", port=API_PORT)


# =========================
# TELEGRAM
# =========================

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Bot ready")


async def handle(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = update.message.text

    await update.message.chat.send_action("typing")

    try:
        reply = await generate_async(text)
    except Exception as e:
        traceback.print_exc()
        reply = f"Error: {e}"

    await update.message.reply_text(reply)


# =========================
# MAIN
# =========================

def main():
    print("⚙️ Entering main()")

    if TOKEN == "PASTE_YOUR_TELEGRAM_BOT_TOKEN_HERE":
        print("❌ SET YOUR TELEGRAM TOKEN")
        input("Press Enter to exit...")
        return

    load_model()

    print("🤖 Starting Telegram in background...")

    def run_telegram():
        try:
            app = Application.builder().token(TOKEN).build()

            app.add_handler(CommandHandler("start", start))
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

            app.run_polling()

        except Exception as e:
            print("🔥 Telegram crashed:", e)

    threading.Thread(target=run_telegram).start()

    print(f"🌐 Starting FastAPI on port {API_PORT} (BLOCKING)")

    # ✅ IMPORTANT: this blocks and keeps Python alive
    uvicorn.run(APP_FASTAPI, host="0.0.0.0", port=API_PORT)
# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    print("✅ __main__ reached")
    main()