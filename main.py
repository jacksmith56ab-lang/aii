import os
import asyncio

from telethon import TelegramClient, events
from telethon.sessions import StringSession
from groq import Groq

API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]
SESSION_STRING = os.environ["SESSION_STRING"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

client = TelegramClient(
    StringSession(SESSION_STRING),
    API_ID,
    API_HASH,
    connection_retries=5,
    retry_delay=1
)

groq = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Отвечай естественно, кратко и по делу, как обычный человек "
    "в личной переписке. Не упоминай, что ты ИИ."
)


async def generate_reply(text: str) -> str:
    try:
        response = await asyncio.to_thread(
            groq.chat.completions.create,
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.7,
            max_tokens=300
        )

        answer = response.choices[0].message.content

        if answer:
            return answer.strip()

    except Exception as e:
        print(
            f"Groq error: {type(e).__name__}: {e}",
            flush=True
        )

    return ""


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

    answer = await generate_reply(text)

    if not answer:
        return

    try:
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


@client.on(events.NewMessage(incoming=True))
async def handler(event):

    if not event.is_private:
        return

    if event.out:
        return

    asyncio.create_task(
        process_message(event)
    )


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
        "Groq AI auto-reply is running.",
        flush=True
    )

    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
