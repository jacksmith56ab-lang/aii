import os
import io
import asyncio
import base64
import tempfile
import json
from urllib.parse import quote

import aiohttp
from PIL import Image
import cv2

from telethon import TelegramClient, events
from telethon.sessions import StringSession
from groq import Groq


# =========================================================
# НАСТРОЙКИ
# =========================================================

API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]
SESSION_STRING = os.environ["SESSION_STRING"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

# Файл со списком игнорируемых пользователей
IGNORE_FILE = "ignored_users.json"

# Сколько сообщений помнить в одном диалоге
MAX_HISTORY = 20


# =========================================================
# TELEGRAM
# =========================================================

client = TelegramClient(
    StringSession(SESSION_STRING),
    API_ID,
    API_HASH,
    connection_retries=5,
    retry_delay=2
)


# =========================================================
# GROQ
# =========================================================

groq = Groq(api_key=GROQ_API_KEY)

TEXT_MODEL = "openai/gpt-oss-20b"

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

WHISPER_MODEL = "whisper-large-v3-turbo"


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    """
Ты обычный человек, который общается в личной переписке Telegram.

Отвечай естественно, живо и по делу.
Не говори, что ты ИИ, бот, языковая модель или программа,
если тебя прямо не спросили об этом.

Не пиши слишком длинные ответы без необходимости.
Используй обычный стиль переписки.

Если пользователь присылает фото, видео, стикер или голосовое,
анализируй его содержимое и отвечай по смыслу.

Если пользователь спрашивает о том, что было сказано ранее,
используй контекст предыдущей переписки.
"""
)


# =========================================================
# ПАМЯТЬ
# =========================================================

chat_history = {}


def get_history(sender_id):
    if sender_id not in chat_history:
        chat_history[sender_id] = []

    return chat_history[sender_id]


def trim_history(history):
    if len(history) > MAX_HISTORY:
        del history[:-MAX_HISTORY]


# =========================================================
# IGNORE SYSTEM
# =========================================================

def load_ignored_users():
    try:
        if not os.path.exists(IGNORE_FILE):
            return set()

        with open(
            IGNORE_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        return set(int(x) for x in data)

    except Exception as e:

        print(
            f"Ignore file read error: {type(e).__name__}: {e}",
            flush=True
        )

        return set()


ignored_users = load_ignored_users()


def save_ignored_users():
    try:

        with open(
            IGNORE_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                sorted(list(ignored_users)),
                f,
                ensure_ascii=False,
                indent=2
            )

    except Exception as e:

        print(
            f"Ignore file save error: {type(e).__name__}: {e}",
            flush=True
        )


def is_ignored(user_id):
    return user_id in ignored_users


# =========================================================
# AI TEXT
# =========================================================

async def ask_groq(sender_id, text):

    history = get_history(sender_id)

    history.append({
        "role": "user",
        "content": text
    })

    trim_history(history)

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

        history.append({
            "role": "assistant",
            "content": answer
        })

        trim_history(history)

        return answer

    except Exception as e:

        print(
            f"Groq text error: {type(e).__name__}: {e}",
            flush=True
        )

        return ""


# =========================================================
# IMAGE -> DATA URL
# =========================================================

def image_to_data_url(image_bytes):

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        if image.mode != "RGB":
            image = image.convert("RGB")

        max_size = 1600

        if max(image.size) > max_size:
            image.thumbnail(
                (max_size, max_size)
            )

        output = io.BytesIO()

        image.save(
            output,
            format="JPEG",
            quality=85
        )

        encoded = base64.b64encode(
            output.getvalue()
        ).decode("utf-8")

        return (
            "data:image/jpeg;base64,"
            + encoded
        )

    except Exception as e:

        print(
            f"Image conversion error: {type(e).__name__}: {e}",
            flush=True
        )

        return None


# =========================================================
# VISION
# =========================================================

async def ask_vision(
    sender_id,
    text,
    image_bytes_list
):

    history = get_history(sender_id)

    content = []

    if text:

        content.append({
            "type": "text",
            "text": text
        })

    else:

        content.append({
            "type": "text",
            "text": (
                "Проанализируй это изображение "
                "и расскажи, что на нём."
            )
        })

    for image_bytes in image_bytes_list[:3]:

        data_url = image_to_data_url(
            image_bytes
        )

        if data_url:

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": data_url
                }
            })

    if len(content) == 1:
        return ""

    history.append({
        "role": "user",
        "content": (
            text
            if text
            else "[пользователь отправил изображение]"
        )
    })

    trim_history(history)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    messages.extend(history[:-1])

    messages.append({
        "role": "user",
        "content": content
    })

    try:

        response = await asyncio.to_thread(
            groq.chat.completions.create,
            model=VISION_MODEL,
            messages=messages,
            temperature=0.6,
            max_tokens=500
        )

        answer = response.choices[0].message.content

        if not answer:
            return ""

        answer = answer.strip()

        history.append({
            "role": "assistant",
            "content": answer
        })

        trim_history(history)

        return answer

    except Exception as e:

        print(
            f"Groq vision error: {type(e).__name__}: {e}",
            flush=True
        )

        return ""


# =========================================================
# VOICE
# =========================================================

async def transcribe_voice(audio_bytes):

    try:

        audio_file = io.BytesIO(
            audio_bytes
        )

        audio_file.name = "voice.ogg"

        result = await asyncio.to_thread(
            groq.audio.transcriptions.create,
            file=audio_file,
            model=WHISPER_MODEL,
            response_format="text"
        )

        if isinstance(result, str):
            return result.strip()

        if hasattr(result, "text"):
            return result.text.strip()

        return ""

    except Exception as e:

        print(
            f"Whisper error: {type(e).__name__}: {e}",
            flush=True
        )

        return ""


# =========================================================
# VIDEO -> FRAMES
# =========================================================

def extract_video_frames(
    video_bytes,
    max_frames=4
):

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False
        ) as temp:

            temp.write(video_bytes)
            temp_path = temp.name

        cap = cv2.VideoCapture(
            temp_path
        )

        if not cap.isOpened():
            return []

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if total_frames <= 0:

            cap.release()
            return []

        positions = []

        if total_frames <= max_frames:

            positions = list(
                range(total_frames)
            )

        else:

            for i in range(max_frames):

                position = int(
                    i
                    * (total_frames - 1)
                    / (max_frames - 1)
                )

                positions.append(position)

        frames = []

        for position in positions:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                position
            )

            success, frame = cap.read()

            if not success:
                continue

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(
                frame
            )

            output = io.BytesIO()

            image.save(
                output,
                format="JPEG",
                quality=80
            )

            frames.append(
                output.getvalue()
            )

        cap.release()

        return frames

    except Exception as e:

        print(
            f"Video processing error: {type(e).__name__}: {e}",
            flush=True
        )

        return []

    finally:

        if temp_path:

            try:
                os.remove(temp_path)
            except Exception:
                pass


# =========================================================
# IMAGE GENERATION
# =========================================================

GENERATION_WORDS = [
    "сгенерируй",
    "сгенерировать",
    "создай",
    "создать",
    "нарисуй",
    "нарисовать",
    "сделай картинку",
    "сделай фото",
    "создай картинку",
    "создай фотографию",
    "сделай изображение",
    "generate",
    "create an image",
    "draw"
]


def is_generation_request(text):

    lower = text.lower().strip()

    return any(
        word in lower
        for word in GENERATION_WORDS
    )


def clean_generation_prompt(text):

    prompt = text.strip()

    replacements = [
        "сгенерируй",
        "сгенерировать",
        "создай",
        "создать",
        "нарисуй",
        "нарисовать",
        "сделай картинку",
        "сделай фото",
        "создай картинку",
        "создай фотографию",
        "сделай изображение",
        "generate",
        "create an image",
        "draw"
    ]

    lower = prompt.lower()

    for word in replacements:

        if lower.startswith(word):

            prompt = prompt[
                len(word):
            ].strip()

            break

    return prompt


async def generate_image(prompt):

    try:

        if not prompt:

            prompt = (
                "A realistic high quality "
                "cinematic photograph"
            )

        encoded_prompt = quote(
            prompt,
            safe=""
        )

        url = (
            "https://image.pollinations.ai/prompt/"
            f"{encoded_prompt}"
            "?width=1024"
            "&height=1024"
            "&nologo=true"
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

                data = await response.read()

                if not data:
                    return None

                return data

    except Exception as e:

        print(
            f"Image generation error: {type(e).__name__}: {e}",
            flush=True
        )

        return None


# =========================================================
# TEXT PROCESSING
# =========================================================

async def process_text(
    event,
    text
):

    sender_id = event.sender_id

    if not text:
        return

    # -----------------------------------------
    # GENERATE IMAGE
    # -----------------------------------------

    if is_generation_request(text):

        prompt = clean_generation_prompt(
            text
        )

        print(
            f"Image generation request from "
            f"{sender_id}: {prompt}",
            flush=True
        )

        await event.respond(
            "🎨 Генерирую..."
        )

        image = await generate_image(
            prompt
        )

        if image:

            file = io.BytesIO(
                image
            )

            file.name = "generated.png"

            await event.client.send_file(
                event.chat_id,
                file,
                caption="🎨 Готово"
            )

            print(
                f"Generated image sent to "
                f"{sender_id}",
                flush=True
            )

        else:

            await event.respond(
                "Не получилось сгенерировать "
                "изображение. Попробуй ещё раз "
                "немного позже."
            )

        return

    # -----------------------------------------
    # NORMAL TEXT
    # -----------------------------------------

    answer = await ask_groq(
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
            f"Reply sent to {sender_id}: "
            f"{answer}",
            flush=True
        )

    except Exception as e:

        print(
            f"Telegram reply error: "
            f"{type(e).__name__}: {e}",
            flush=True
        )


# =========================================================
# PHOTO
# =========================================================

async def process_photo(event):

    sender_id = event.sender_id

    print(
        f"Downloading photo from "
        f"{sender_id}...",
        flush=True
    )

    try:

        data = await event.download_media(
            file=bytes
        )

        if not data:
            return

        caption = (
            event.raw_text or ""
        ).strip()

        answer = await ask_vision(
            sender_id,
            caption,
            [data]
        )

        if answer:

            await event.reply(
                answer
            )

            print(
                f"Vision reply sent to "
                f"{sender_id}: {answer}",
                flush=True
            )

    except Exception as e:

        print(
            f"Photo processing error: "
            f"{type(e).__name__}: {e}",
            flush=True
        )


# =========================================================
# VIDEO
# =========================================================

async def process_video(event):

    sender_id = event.sender_id

    print(
        f"Downloading video from "
        f"{sender_id}...",
        flush=True
    )

    try:

        data = await event.download_media(
            file=bytes
        )

        if not data:
            return

        frames = await asyncio.to_thread(
            extract_video_frames,
            data,
            4
        )

        if not frames:

            await event.reply(
                "Не смог прочитать это видео."
            )

            return

        caption = (
            event.raw_text or ""
        ).strip()

        if not caption:

            caption = (
                "Посмотри кадры этого видео "
                "и расскажи, что в нём "
                "происходит."
            )

        answer = await ask_vision(
            sender_id,
            caption,
            frames
        )

        if answer:

            await event.reply(
                answer
            )

    except Exception as e:

        print(
            f"Video processing error: "
            f"{type(e).__name__}: {e}",
            flush=True
        )


# =========================================================
# VOICE
# =========================================================

async def process_voice(event):

    sender_id = event.sender_id

    print(
        f"Downloading voice from "
        f"{sender_id}...",
        flush=True
    )

    try:

        data = await event.download_media(
            file=bytes
        )

        if not data:
            return

        text = await transcribe_voice(
            data
        )

        if not text:

            await event.reply(
                "Не получилось разобрать "
                "голосовое."
            )

            return

        print(
            f"Voice from {sender_id}: "
            f"{text}",
            flush=True
        )

        answer = await ask_groq(
            sender_id,
            f"[Голосовое сообщение]\n{text}"
        )

        if answer:

            await event.reply(
                answer
            )

    except Exception as e:

        print(
            f"Voice processing error: "
            f"{type(e).__name__}: {e}",
            flush=True
        )


# =========================================================
# STICKER
# =========================================================

async def process_sticker(event):

    sender_id = event.sender_id

    try:

        data = await event.download_media(
            file=bytes
        )

        if not data:
            return

        try:

            image = Image.open(
                io.BytesIO(data)
            )

            output = io.BytesIO()

            image.convert(
                "RGBA"
            ).save(
                output,
                format="PNG"
            )

            image_bytes = (
                output.getvalue()
            )

            answer = await ask_vision(
                sender_id,
                (
                    "Что изображено на этом "
                    "стикере? Опиши его и "
                    "объясни, какую эмоцию "
                    "или смысл он передаёт."
                ),
                [image_bytes]
            )

            if answer:

                await event.reply(
                    answer
                )

            return

        except Exception:
            pass

        await event.reply(
            "Это анимированный или "
            "видео-стикер. Такой формат "
            "пока не могу нормально разобрать."
        )

    except Exception as e:

        print(
            f"Sticker processing error: "
            f"{type(e).__name__}: {e}",
            flush=True
        )


# =========================================================
# COMMANDS
# =========================================================

async def handle_command(event, text):

    # Только владелец аккаунта
    me = await client.get_me()

    if event.sender_id != me.id:
        return False

    command = text.strip()

    # -----------------------------------------
    # /ignore
    # -----------------------------------------

    if command == "/ignore":

        # Игнорируем текущего собеседника
        target_id = event.chat_id

        if target_id == me.id:

            await event.reply(
                "Нельзя добавить самого себя."
            )

            return True

        ignored_users.add(
            int(target_id)
        )

        save_ignored_users()

        await event.reply(
            f"🔕 Пользователь "
            f"{target_id} добавлен в игнор."
        )

        print(
            f"Added to ignore: {target_id}",
            flush=True
        )

        return True

    # -----------------------------------------
    # /unignore
    # -----------------------------------------

    if command == "/unignore":

        target_id = event.chat_id

        if target_id in ignored_users:

            ignored_users.remove(
                target_id
            )

            save_ignored_users()

            await event.reply(
                f"🔔 Пользователь "
                f"{target_id} убран из игнора."
            )

        else:

            await event.reply(
                "Этот пользователь "
                "не находится в игноре."
            )

        return True

    # -----------------------------------------
    # /ignore ID
    # -----------------------------------------

    if command.startswith("/ignore "):

        value = command[
            len("/ignore "):
        ].strip()

        try:

            target_id = int(value)

            if target_id == me.id:

                await event.reply(
                    "Нельзя добавить "
                    "самого себя."
                )

                return True

            ignored_users.add(
                target_id
            )

            save_ignored_users()

            await event.reply(
                f"🔕 Пользователь "
                f"{target_id} добавлен в игнор."
            )

        except ValueError:

            await event.reply(
                "Использование:\n"
                "/ignore 123456789"
            )

        return True

    # -----------------------------------------
    # /unignore ID
    # -----------------------------------------

    if command.startswith("/unignore "):

        value = command[
            len("/unignore "):
        ].strip()

        try:

            target_id = int(value)

            if target_id in ignored_users:

                ignored_users.remove(
                    target_id
                )

                save_ignored_users()

                await event.reply(
                    f"🔔 Пользователь "
                    f"{target_id} убран из игнора."
                )

            else:

                await event.reply(
                    "Этого пользователя "
                    "нет в игноре."
                )

        except ValueError:

            await event.reply(
                "Использование:\n"
                "/unignore 123456789"
            )

        return True

    # -----------------------------------------
    # /ignored
    # -----------------------------------------

    if command == "/ignored":

        if not ignored_users:

            await event.reply(
                "📋 Список игнора пуст."
            )

        else:

            lines = [
                "🔕 Пользователи в игноре:"
            ]

            for user_id in sorted(
                ignored_users
            ):

                lines.append(
                    str(user_id)
                )

            await event.reply(
                "\n".join(lines)
            )

        return True

    # -----------------------------------------
    # /ignoreme
    # -----------------------------------------

    if command == "/ignoreme":

        target_id = event.chat_id

        if target_id == me.id:

            await event.reply(
                "Нельзя добавить самого себя."
            )

            return True

        ignored_users.add(
            int(target_id)
        )

        save_ignored_users()

        await event.reply(
            "🔕 Этот пользователь добавлен "
            "в игнор."
        )

        return True

    # -----------------------------------------
    # /unignoreme
    # -----------------------------------------

    if command == "/unignoreme":

        target_id = event.chat_id

        if target_id in ignored_users:

            ignored_users.remove(
                target_id
            )

            save_ignored_users()

            await event.reply(
                "🔔 Игнор снят."
            )

        else:

            await event.reply(
                "Этот пользователь "
                "не находится в игноре."
            )

        return True

    return False


# =========================================================
# TELEGRAM HANDLER
# =========================================================

@client.on(
    events.NewMessage(
        incoming=True
    )
)
async def handler(event):

    # Только личные сообщения
    if not event.is_private:
        return

    # Не отвечаем сами себе
    if event.out:
        return

    try:

        message = event.message

        text = (
            event.raw_text or ""
        ).strip()

        # =====================================
        # КОМАНДЫ ВЛАДЕЛЬЦА
        # =====================================

        if text.startswith("/"):

            handled = await handle_command(
                event,
                text
            )

            if handled:
                return

        # =====================================
        # IGNORE
        # =====================================

        sender_id = event.sender_id

        if is_ignored(sender_id):

            print(
                f"Ignored message from "
                f"{sender_id}",
                flush=True
            )

            return

        # =====================================
        # VOICE
        # =====================================

        if message.voice:

            asyncio.create_task(
                process_voice(event)
            )

            return

        # =====================================
        # VIDEO
        # =====================================

        if message.video:

            asyncio.create_task(
                process_video(event)
            )

            return

        # =====================================
        # PHOTO
        # =====================================

        if message.photo:

            asyncio.create_task(
                process_photo(event)
            )

            return

        # =====================================
        # STICKER
        # =====================================

        if message.sticker:

            asyncio.create_task(
                process_sticker(event)
            )

            return

        # =====================================
        # TEXT
        # =====================================

        if text:

            print(
                f"Incoming message from "
                f"{sender_id}: {text}",
                flush=True
            )

            asyncio.create_task(
                process_text(
                    event,
                    text
                )
            )

    except Exception as e:

        print(
            f"Handler error: "
            f"{type(e).__name__}: {e}",
            flush=True
        )


# =========================================================
# MAIN
# =========================================================

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
        f"Ignored users: "
        f"{len(ignored_users)}",
        flush=True
    )

    print(
        "AI auto-reply is running.",
        flush=True
    )

    print(
        "Enabled: text, memory, photos, "
        "video, voice, stickers, "
        "image generation, ignore list.",
        flush=True
    )

    await client.run_until_disconnected()


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    asyncio.run(main())
