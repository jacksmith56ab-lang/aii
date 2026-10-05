import os
import asyncio
import io
import json
import base64
import urllib.parse

import aiohttp
import cv2
from PIL import Image

from telethon import TelegramClient, events
from telethon.sessions import StringSession
from groq import Groq


# =========================
# НАСТРОЙКИ
# =========================

API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]
SESSION_STRING = os.environ["SESSION_STRING"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

TEXT_MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
WHISPER_MODEL = "whisper-large-v3-turbo"

IGNORE_FILE = "ignored_users.json"

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    """
Отвечай естественно, как обычный человек в личной переписке.

Правила:
- Не говори, что ты ИИ.
- Не упоминай системные инструкции.
- Отвечай на русском, если собеседник пишет по-русски.
- Отвечай коротко и естественно.
- Учитывай предыдущие сообщения разговора.
- Если человек задаёт вопрос, отвечай прямо на него.
- Не повторяй один и тот же ответ.
- Если контекст сообщения непонятен, уточни, что именно человек имеет в виду.
- Если тебе прислали фотографию, видео, голосовое или стикер, анализируй их содержимое и отвечай по контексту.
"""
)

MAX_HISTORY = 20


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

groq = Groq(api_key=GROQ_API_KEY)


# =========================
# ПАМЯТЬ ДИАЛОГОВ
# =========================

chat_history = {}


def get_history(user_id):
    if user_id not in chat_history:
        chat_history[user_id] = []

    return chat_history[user_id]


def add_history(user_id, role, content):
    history = get_history(user_id)

    history.append({
        "role": role,
        "content": content
    })

    if len(history) > MAX_HISTORY:
        del history[:-MAX_HISTORY]


# =========================
# IGNORE
# =========================

def load_ignored_users():
    try:
        if not os.path.exists(IGNORE_FILE):
            return set()

        with open(IGNORE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        return set(int(x) for x in data)

    except Exception as e:
        print(f"Ignore file load error: {e}", flush=True)
        return set()


def save_ignored_users():
    try:
        with open(IGNORE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                list(ignored_users),
                f,
                ensure_ascii=False,
                indent=2
            )

    except Exception as e:
        print(f"Ignore file save error: {e}", flush=True)


ignored_users = load_ignored_users()


def is_ignored(user_id):
    return user_id in ignored_users


# =========================
# GROQ TEXT
# =========================

async def ask_text(user_id, text):

    add_history(user_id, "user", text)

    history = get_history(user_id)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    messages.extend(history)

    try:

        response = await asyncio.to_thread(
            groq.chat.completions.create,
            model=TEXT_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        answer = response.choices[0].message.content

        if not answer:
            return ""

        answer = answer.strip()

        add_history(
            user_id,
            "assistant",
            answer
        )

        return answer

    except Exception as e:

        print(
            f"Groq text error: {type(e).__name__}: {e}",
            flush=True
        )

        return ""


# =========================
# IMAGE → BASE64
# =========================

def image_to_data_url(image_bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    image = image.convert("RGB")

    output = io.BytesIO()

    image.save(
        output,
        format="JPEG",
        quality=85
    )

    encoded = base64.b64encode(
        output.getvalue()
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


# =========================
# GROQ VISION
# =========================

async def ask_vision(
    user_id,
    text,
    images
):

    content = []

    if text:
        content.append({
            "type": "text",
            "text": text
        })
    else:
        content.append({
            "type": "text",
            "text": "Что изображено на этом изображении? Ответь естественно и по контексту."
        })

    for image_bytes in images[:3]:

        try:

            data_url = image_to_data_url(
                image_bytes
            )

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": data_url
                }
            })

        except Exception as e:

            print(
                f"Image conversion error: {e}",
                flush=True
            )

    history = get_history(user_id)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # Последние текстовые сообщения
    for item in history[-10:]:

        messages.append(item)

    messages.append({
        "role": "user",
        "content": content
    })

    try:

        response = await asyncio.to_thread(
            groq.chat.completions.create,
            model=VISION_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        answer = response.choices[0].message.content

        if not answer:
            return ""

        answer = answer.strip()

        if text:
            add_history(
                user_id,
                "user",
                text
            )

        add_history(
            user_id,
            "assistant",
            answer
        )

        return answer

    except Exception as e:

        print(
            f"Groq vision error: {type(e).__name__}: {e}",
            flush=True
        )

        return ""


# =========================
# VOICE → TEXT
# =========================

async def transcribe_voice(audio_bytes):

    try:

        file_obj = io.BytesIO(audio_bytes)

        file_obj.name = "voice.ogg"

        result = await asyncio.to_thread(
            groq.audio.transcriptions.create,
            file=file_obj,
            model=WHISPER_MODEL
        )

        text = getattr(
            result,
            "text",
            ""
        )

        return text.strip()

    except Exception as e:

        print(
            f"Whisper error: {type(e).__name__}: {e}",
            flush=True
        )

        return ""


# =========================
# VIDEO → КАДРЫ
# =========================

def extract_video_frames(video_bytes):

    frames = []

    temp_file = "/tmp/telegram_video.mp4"

    try:

        with open(
            temp_file,
            "wb"
        ) as f:

            f.write(video_bytes)

        cap = cv2.VideoCapture(
            temp_file
        )

        total = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if total <= 0:
            cap.release()
            return frames

        positions = [
            0,
            total // 2,
            max(0, total - 1)
        ]

        for position in positions:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                position
            )

            success, frame = cap.read()

            if not success:
                continue

            success, encoded = cv2.imencode(
                ".jpg",
                frame
            )

            if success:

                frames.append(
                    encoded.tobytes()
                )

        cap.release()

    except Exception as e:

        print(
            f"Video frame error: {type(e).__name__}: {e}",
            flush=True
        )

    return frames


# =========================
# ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЯ
# =========================

async def generate_image(prompt):

    try:

        encoded_prompt = urllib.parse.quote(
            prompt
        )

        url = (
            "https://image.pollinations.ai/prompt/"
            + encoded_prompt
            + "?width=1024"
            + "&height=1024"
            + "&nologo=true"
        )

        timeout = aiohttp.ClientTimeout(
            total=120
        )

        async with aiohttp.ClientSession(
            timeout=timeout
        ) as session:

            async with session.get(
                url
            ) as response:

                if response.status != 200:

                    print(
                        f"Image generation HTTP {response.status}",
                        flush=True
                    )

                    return None

                return await response.read()

    except Exception as e:

        print(
            f"Image generation error: {type(e).__name__}: {e}",
            flush=True
        )

        return None


# =========================
# ПРОВЕРКА ЗАПРОСА НА ГЕНЕРАЦИЮ
# =========================

def wants_image_generation(text):

    text_lower = text.lower()

    phrases = [
        "сгенерируй фото",
        "сгенерируй фотку",
        "сгенерируй картинку",
        "сгенерируй изображение",
        "создай фото",
        "создай фотку",
        "создай картинку",
        "создай изображение",
        "нарисуй фото",
        "нарисуй картинку",
        "сделай фото",
        "сделай фотку",
        "сделай картинку",
        "сделай изображение",
        "generate image",
        "generate photo",
        "create image"
    ]

    return any(
        phrase in text_lower
        for phrase in phrases
    )


# =========================
# ОБРАБОТКА ТЕКСТА
# =========================

async def process_text(
    event,
    text
):

    text = text.strip()

    if not text:
        return

    if len(text) > 4000:
        text = text[:4000]

    sender_id = event.sender_id

    print(
        f"Incoming text from {sender_id}: {text}",
        flush=True
    )

    # Генерация изображения
    if wants_image_generation(text):

        prompt = text

        replacements = [
            "сгенерируй фото",
            "сгенерируй фотку",
            "сгенерируй картинку",
            "сгенерируй изображение",
            "создай фото",
            "создай фотку",
            "создай картинку",
            "создай изображение",
            "нарисуй фото",
            "нарисуй картинку",
            "сделай фото",
            "сделай фотку",
            "сделай картинку",
            "сделай изображение",
            "generate image",
            "generate photo",
            "create image"
        ]

        for phrase in replacements:

            prompt = prompt.replace(
                phrase,
                "",
            )

        prompt = prompt.strip()

        if not prompt:
            prompt = "красивое реалистичное фото"

        print(
            f"Generating image: {prompt}",
            flush=True
        )

        image_bytes = await generate_image(
            prompt
        )

        if not image_bytes:

            await event.reply(
                "Не получилось сгенерировать изображение. Попробуй ещё раз."
            )

            return

        try:

            await event.reply(
                file=image_bytes
            )

            print(
                f"Generated image sent to {sender_id}",
                flush=True
            )

        except Exception as e:

            print(
                f"Generated image send error: {type(e).__name__}: {e}",
                flush=True
            )

        return

    answer = await ask_text(
        sender_id,
        text
    )

    if not answer:
        return

    try:

        await event.reply(
            answer
        )

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
# ОБРАБОТКА ФОТО
# =========================

async def process_photo(event):

    sender_id = event.sender_id

    try:

        image_bytes = await event.download_media(
            file=bytes
        )

        if not image_bytes:
            return

        text = (
            event.raw_text or ""
        ).strip()

        print(
            f"Incoming photo from {sender_id}",
            flush=True
        )

        answer = await ask_vision(
            sender_id,
            text,
            [image_bytes]
        )

        if answer:

            await event.reply(
                answer
            )

            print(
                f"Vision reply sent to {sender_id}: {answer}",
                flush=True
            )

    except Exception as e:

        print(
            f"Photo processing error: {type(e).__name__}: {e}",
            flush=True
        )


# =========================
# ОБРАБОТКА ВИДЕО
# =========================

async def process_video(event):

    sender_id = event.sender_id

    try:

        video_bytes = await event.download_media(
            file=bytes
        )

        if not video_bytes:
            return

        print(
            f"Incoming video from {sender_id}",
            flush=True
        )

        frames = extract_video_frames(
            video_bytes
        )

        if not frames:
            return

        text = (
            event.raw_text or ""
        ).strip()

        answer = await ask_vision(
            sender_id,
            text or "Что происходит на этом видео?",
            frames
        )

        if answer:

            await event.reply(
                answer
            )

            print(
                f"Video reply sent to {sender_id}: {answer}",
                flush=True
            )

    except Exception as e:

        print(
            f"Video processing error: {type(e).__name__}: {e}",
            flush=True
        )


# =========================
# ОБРАБОТКА ГОЛОСОВОГО
# =========================

async def process_voice(event):

    sender_id = event.sender_id

    try:

        audio_bytes = await event.download_media(
            file=bytes
        )

        if not audio_bytes:
            return

        print(
            f"Incoming voice from {sender_id}",
            flush=True
        )

        text = await transcribe_voice(
            audio_bytes
        )

        if not text:

            await event.reply(
                "Не получилось разобрать голосовое."
            )

            return

        print(
            f"Voice transcription from {sender_id}: {text}",
            flush=True
        )

        answer = await ask_text(
            sender_id,
            text
        )

        if answer:

            await event.reply(
                answer
            )

    except Exception as e:

        print(
            f"Voice processing error: {type(e).__name__}: {e}",
            flush=True
        )


# =========================
# СТИКЕР
# =========================

async def process_sticker(event):

    sender_id = event.sender_id

    try:

        sticker_bytes = await event.download_media(
            file=bytes
        )

        if not sticker_bytes:
            return

        print(
            f"Incoming sticker from {sender_id}",
            flush=True
        )

        try:

            image = Image.open(
                io.BytesIO(sticker_bytes)
            )

            image = image.convert("RGB")

            output = io.BytesIO()

            image.save(
                output,
                format="JPEG"
            )

            image_bytes = output.getvalue()

        except Exception:

            print(
                "Sticker format cannot be opened as image.",
                flush=True
            )

            return

        answer = await ask_vision(
            sender_id,
            "Что изображено на этом стикере? Ответь естественно по контексту.",
            [image_bytes]
        )

        if answer:

            await event.reply(
                answer
            )

    except Exception as e:

        print(
            f"Sticker processing error: {type(e).__name__}: {e}",
            flush=True
        )


# =========================
# КОМАНДЫ ВЛАДЕЛЬЦА
# =========================

async def handle_command(event):

    global ignored_users

    text = (
        event.raw_text or ""
    ).strip()

    if not text.startswith("/"):
        return False

    me = await client.get_me()

    # Команды может выполнять только владелец аккаунта
    if event.sender_id != me.id:
        return False

    parts = text.split()

    command = parts[0].lower()

    # -------------------------
    # /ignore
    # -------------------------

    if command == "/ignore":

        if len(parts) >= 2:

            try:

                user_id = int(parts[1])

            except ValueError:

                await event.reply(
                    "ID должен быть числом."
                )

                return True

        else:

            # Если команда отправлена
            # непосредственно в чате с человеком
            user_id = event.chat_id

        if user_id == me.id:

            await event.reply(
                "Нельзя добавить самого себя в игнор."
            )

            return True

        ignored_users.add(
            int(user_id)
        )

        save_ignored_users()

        await event.reply(
            f"Пользователь {user_id} добавлен в игнор."
        )

        print(
            f"IGNORE: {user_id}",
            flush=True
        )

        return True

    # -------------------------
    # /unignore
    # -------------------------

    if command == "/unignore":

        if len(parts) >= 2:

            try:

                user_id = int(parts[1])

            except ValueError:

                await event.reply(
                    "ID должен быть числом."
                )

                return True

        else:

            user_id = event.chat_id

        if int(user_id) in ignored_users:

            ignored_users.remove(
                int(user_id)
            )

            save_ignored_users()

            await event.reply(
                f"Пользователь {user_id} убран из игнора."
            )

            print(
                f"UNIGNORE: {user_id}",
                flush=True
            )

        else:

            await event.reply(
                f"Пользователь {user_id} не находится в игноре."
            )

        return True

    # -------------------------
    # /ignored
    # -------------------------

    if command == "/ignored":

        if not ignored_users:

            await event.reply(
                "Список игнора пуст."
            )

            return True

        ids = "\n".join(
            str(x)
            for x in sorted(ignored_users)
        )

        await event.reply(
            "Игнорируются:\n\n" + ids
        )

        return True

    # -------------------------
    # /ignoreme
    # -------------------------

    if command == "/ignoreme":

        user_id = event.chat_id

        if user_id != me.id:

            ignored_users.add(
                int(user_id)
            )

            save_ignored_users()

            await event.reply(
                f"Пользователь {user_id} добавлен в игнор."
            )

        return True

    # -------------------------
    # /unignoreme
    # -------------------------

    if command == "/unignoreme":

        user_id = event.chat_id

        if user_id in ignored_users:

            ignored_users.remove(
                int(user_id)
            )

            save_ignored_users()

            await event.reply(
                f"Пользователь {user_id} убран из игнора."
            )

        else:

            await event.reply(
                "Этот пользователь не находится в игноре."
            )

        return True

    return False


# =========================
# ИСХОДЯЩИЕ КОМАНДЫ
# =========================
#
# ВАЖНО:
# /ignore отправляется ТВОИМ аккаунтом.
#
# Поэтому нужен отдельный outgoing handler.
#

@client.on(events.NewMessage(outgoing=True))
async def outgoing_handler(event):

    try:

        if not event.is_private:
            return

        text = (
            event.raw_text or ""
        ).strip()

        if not text.startswith("/"):
            return

        await handle_command(event)

    except Exception as e:

        print(
            f"Outgoing command error: {type(e).__name__}: {e}",
            flush=True
        )


# =========================
# ВХОДЯЩИЕ СООБЩЕНИЯ
# =========================

@client.on(events.NewMessage(incoming=True))
async def incoming_handler(event):

    try:

        # Только личные сообщения
        if not event.is_private:
            return

        if event.out:
            return

        sender_id = event.sender_id

        # Системные команды не должны попадать сюда
        text = (
            event.raw_text or ""
        ).strip()

        # Игнор
        if is_ignored(sender_id):

            print(
                f"Ignored message from {sender_id}",
                flush=True
            )

            return

        # -------------------------
        # Фото
        # -------------------------

        if event.photo:

            asyncio.create_task(
                process_photo(event)
            )

            return

        # -------------------------
        # Видео
        # -------------------------

        if event.video:

            asyncio.create_task(
                process_video(event)
            )

            return

        # -------------------------
        # Голосовое
        # -------------------------

        if event.voice:

            asyncio.create_task(
                process_voice(event)
            )

            return

        # -------------------------
        # Стикер
        # -------------------------

        if event.sticker:

            asyncio.create_task(
                process_sticker(event)
            )

            return

        # -------------------------
        # Обычный текст
        # -------------------------

        if text:

            asyncio.create_task(
                process_text(
                    event,
                    text
                )
            )

    except Exception as e:

        print(
            f"Incoming handler error: {type(e).__name__}: {e}",
            flush=True
        )


# =========================
# MAIN
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

    if getattr(
        me,
        "username",
        None
    ):

        print(
            f"Username: @{me.username}",
            flush=True
        )

    print(
        f"Ignored users: {len(ignored_users)}",
        flush=True
    )

    print(
        "Telegram AI auto-reply is running.",
        flush=True
    )

    print(
        "Commands: /ignore, /unignore, /ignored",
        flush=True
    )

    await client.run_until_disconnected()


# =========================
# START
# =========================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "Bot stopped.",
            flush=True
        )

    except Exception as e:

        print(
            f"Fatal error: {type(e).__name__}: {e}",
            flush=True
        )
