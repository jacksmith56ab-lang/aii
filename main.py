import os
import asyncio

from telethon import TelegramClient, events
from telethon.sessions import StringSession
from google import genai


# =========================
# НАСТРОЙКИ
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
    API_HASH
)


# =========================
# GEMINI
# =========================

gemini = genai.Client(
    api_key=GEMINI_API_KEY
)


# =========================
# СИСТЕМНЫЙ ПРОМПТ
# =========================

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    """
Отвечай естественно и кратко, как обычный человек в личной переписке.

Правила:
- Не говори, что ты ИИ, если тебя прямо об этом не спрашивают.
- Отвечай по смыслу сообщения.
- Не используй слишком официальный стиль.
- Не пиши длинные ответы без необходимости.
- Не начинай каждый ответ одинаково.
- Если собеседник пишет на русском, отвечай на русском.
- Не добавляй лишние объяснения.
""".strip()
)


# =========================
# ЗАПРОС К GEMINI
# =========================

async def generate_reply(user_message: str) -> str:

    prompt = f"""
{SYSTEM_PROMPT}

Сообщение собеседника:
{user_message}
""".strip()

    response = await asyncio.to_thread(
        gemini.models.generate_content,
        model="gemini-3.8-flash",
        contents=prompt
    )

    answer = getattr(response, "text", None)

    if not answer:
        return ""

    return answer.strip()


# =========================
# ОБРАБОТКА СООБЩЕНИЙ
# =========================

@client.on(events.NewMessage(incoming=True))
async def handler(event):

    # Только личные сообщения
    if not event.is_private:
        return

    # Не отвечаем на свои сообщения
    if event.out:
        return

    text = (event.raw_text or "").strip()

    # Игнорируем пустые сообщения
    if not text:
        return

    try:

        print(
            f"Incoming message from {event.sender_id}: {text}",
            flush=True
        )

        # Запрашиваем ответ у Gemini
        answer = await generate_reply(text)

        if not answer:
            print(
                "Gemini returned an empty response.",
                flush=True
            )
            return

        # Небольшая задержка перед ответом
        await asyncio.sleep(1.5)

        # Отправляем ответ
        await event.reply(answer)

        print(
            f"Reply sent to {event.sender_id}: {answer}",
            flush=True
        )

    except Exception as e:

        print(
            f"Handler error: {type(e).__name__}: {e}",
            flush=True
        )


# =========================
# ЗАПУСК
# =========================

async def main():

    print(
        "Starting Telegram AI auto-reply...",
        flush=True
    )

    await client.connect()

    # Проверяем Session String
    if not await client.is_user_authorized():

        raise RuntimeError(
            "SESSION_STRING is missing or invalid."
        )

    # Получаем данные аккаунта
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

    # Оставляем Telegram подключённым
    await client.run_until_disconnected()


# =========================
# START
# =========================

if __name__ == "__main__":
    asyncio.run(main())