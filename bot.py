print("🚀 Starting DeepSeek Telegram Bot...")
print("SCRIPT_VERSION = deepseek_telegram_v3_after_think_v1")

# =========================
# IMPORTS
# =========================

import asyncio
import traceback
import requests
import html

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes


# =========================
# CONFIG
# =========================

TOKEN = "ADD TOKEN"

# Your working endpoint from curl
DEEPSEEK_URL = "http://127.0.0.1:9000/v3/chat/completions"

# Your working model name from curl
DEEPSEEK_MODEL = "deepseek"

REQUEST_TIMEOUT_SECONDS = 180
MAX_COMPLETION_TOKENS = 512


# =========================
# CLEAN RESPONSE
# =========================

def clean_after_think_tag(text):
    """
    Keep only the text AFTER </think> or </thinking>.

    Handles your actual response format:
    &lt;/think&gt;

    Example:
    "reasoning... &lt;/think&gt;\\n\\nHello!"
    becomes:
    "Hello!"
    """

    if text is None:
        return ""

    output = str(text)

    # Convert HTML entities:
    # &lt;/think&gt; -> </think>
    # &lt;/thinking&gt; -> </thinking>
    output = html.unescape(output)

    # Take everything after </think>
    if "</think>" in output:
        output = output.split("</think>", 1)[1]

    # Or take everything after </thinking>
    elif "</thinking>" in output:
        output = output.split("</thinking>", 1)[1]

    return output.strip()


# =========================
# MODEL CALL
# =========================

def generate(prompt):
    try:
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "max_completion_tokens": MAX_COMPLETION_TOKENS
        }

        response = requests.post(
            DEEPSEEK_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS
        )

        if response.status_code != 200:
            return (
                f"DeepSeek HTTP error {response.status_code}:\n"
                f"{response.text}"
            )

        data = response.json()

        output = data["choices"][0]["message"]["content"]

        return clean_after_think_tag(output)

    except Exception as e:
        traceback.print_exc()
        return f"Error calling DeepSeek: {e}"


async def generate_async(prompt):
    return await asyncio.to_thread(generate, prompt)


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
    parts = split_text_safely(text)

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

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ DeepSeek bot ready.\n\n"
        "Send me a message and I will reply using your local DeepSeek server.",
        parse_mode=None
    )


async def probe(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.chat.send_action("typing")

    reply = await generate_async("Say hello.")

    await send_long(update, reply)


async def debug(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    info = (
        "Debug info:\n"
        "SCRIPT_VERSION = deepseek_telegram_v3_after_think_v1\n"
        f"DEEPSEEK_URL = {DEEPSEEK_URL}\n"
        f"DEEPSEEK_MODEL = {DEEPSEEK_MODEL}\n"
        f"MAX_COMPLETION_TOKENS = {MAX_COMPLETION_TOKENS}"
    )

    await send_long(update, info)


# =========================
# TELEGRAM MESSAGE HANDLER
# =========================

async def handle(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_text = update.message.text.strip()

    if not user_text:
        return

    await update.message.chat.send_action("typing")

    reply = await generate_async(user_text)

    await send_long(update, reply)


# =========================
# MAIN
# =========================

def main():
    print("⚙️ Entering main()")

    if TOKEN == "PASTE_YOUR_TELEGRAM_BOT_TOKEN_HERE":
        print("❌ SET YOUR TELEGRAM TOKEN")
        input("Press Enter to exit...")
        return

    print("✅ Starting Telegram polling...")

    try:
        app = Application.builder().token(TOKEN.strip()).build()

        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("probe", probe))
        app.add_handler(CommandHandler("debug", debug))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

        app.run_polling()

    except Exception:
        print("🔥 Telegram crashed:")
        traceback.print_exc()
        input("Press Enter to exit...")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    print("✅ __main__ reached")
    main()
