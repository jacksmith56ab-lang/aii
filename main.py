import os
import asyncio

from telethon import TelegramClient, events
from telethon.sessions import StringSession
from google import genai


# =========================
# ENV
# =========================

API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]
SESSION_STRING = os.environ["SESSION_STRING"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]


# =========================
# TELEGRAM
# =========================

client = TelegramClient(
    StringSession(SESSION_STRING),
    API_ID,
    API_HASH,
    connection_retries=5,
    retry_delay=1
)


# =========================
# GEMINI
# =========================

gemini = genai.Client(
    api_key=GEMINI_API_KEY
)


SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Отвечай естественно, кратко и по делу, как обычный человек "
    "в личной переписке. Не упоминай, что ты ИИ."
)


# =========================
# GEMINI
# =========================

async def generate_reply(text: str) -> str:

    prompt = f"{SYSTEM_PROMPT}\n\nСообщение:\n{text}"

    try:
        response = await asyncio.to_thread(
            gemini.models.generate_content,
            model="gemini-3.8-flash",
            contents=prompt
        )

        answer = getattr(response, "text", None)

        if answer:
            return answer.strip()

    except Exception as e:

        print(
            f"Gemini error: {type(e).__name__}: {e}",
            flush=True
        )

    return ""


# =========================
# ОБРАБОТКА ОДНОГО СООБЩЕНИЯ
# =========================

async def process_message(event):

    text = (event.raw_text or "").strip()

    if not text:
        return

    if len(text) > 4000:
        text = text[:4000]

    sender_id = event.sender_id

    print(
        f"Incoming message from {sender_id}: {text}",
        flush=True
    )

    # Каждый message получает собственную задачу Gemini
    answer = await generate_reply(text)

    if not answer:
        return

    try:

        # Отправляем сразу после получения ответа
        await event.reply(answer)

        print(
            f"Reply sent to {sender_id}: {answer}",
            flush=True
        )

    except Exception as e:

        print(
            f"Telegram reply error: {type(e).__name__}: {e}",
            flush=True
        )


# =========================
# TELEGRAM HANDLER
# =========================

@client.on(events.NewMessage(incoming=True))
async def handler(event):

    if not event.is_private:
        return

    if event.out:
        return

    # Создаём отдельную asyncio-задачу
    # для КАЖДОГО сообщения.
    asyncio.create_task(
        process_message(event)
    )


# =========================
# START
# =========================

async def main():

    print(
        "Starting Telegram AI auto-reply...",
        flush=True
    )

    await client.connect()

    if not await client.is_user_authorized():

        raise RuntimeError(
            "SESSION_STRING is missing or invalid."
        )

    me = await client.get_me()

    print(
        f"Authorized as user id={me.id}",
        flush=True
    )

    if getattr(me, "username", None):

        print(
            f"Username: @{me.username}",
            flush=True
        )

    print(
        "AI auto-reply is running.",
        flush=True
    )

    await client.run_until_disconnected()


# =========================
# RUN
# =========================

if __name__ == "__main__":
    asyncio.run(main())