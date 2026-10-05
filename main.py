import os, asyncio
from telethon import TelegramClient, events
from telethon.sessions import StringSession
from google import genai

API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]
SESSION_STRING = os.environ["SESSION_STRING"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

client = TelegramClient(StringSession(SESSION_STRING), API_ID, API_HASH)
gemini = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Отвечай естественно и кратко, как обычный человек в личной переписке. "
    "Не упоминай, что ты ИИ, если тебя прямо об этом не спрашивают."
)

@client.on(events.NewMessage(incoming=True))
async def handler(event):
    if not event.is_private or event.out:
        return
    text = (event.raw_text or "").strip()
    if not text:
        return
    try:
        response = await asyncio.to_thread(
            gemini.models.generate_content,
            model="gemini-2.5-flash",
            contents=f"{SYSTEM_PROMPT}\n\nСообщение:\n{text}"
        )
        answer = (getattr(response, "text", None) or "").strip()
        if answer:
            await asyncio.sleep(1.5)
            await event.reply(answer)
            print(f"Replied to {event.sender_id}", flush=True)
    except Exception as e:
        print(f"Handler error: {type(e).__name__}: {e}", flush=True)

async def main():
    print("Starting Telegram AI auto-reply...", flush=True)
    await client.connect()
    if not await client.is_user_authorized():
        raise RuntimeError("SESSION_STRING is missing or invalid.")
    me = await client.get_me()
    print(f"Authorized as user id={me.id}", flush=True)
    print("AI auto-reply is running.", flush=True)
    await client.run_until_disconnected()

asyncio.run(main())
