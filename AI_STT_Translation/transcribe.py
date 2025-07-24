import httpx
from urllib.parse import quote
from constants import translation_lang
from fastapi import UploadFile

import numpy as np
from fastrtc import (
    AdditionalOutputs,
    audio_to_bytes,
    get_cloudflare_turn_credentials_async,
)
from groq import AsyncClient

groq_client = AsyncClient()

async def get_credentials(huggingface_token: str):
    return await get_cloudflare_turn_credentials_async(hf_token=huggingface_token)

async def transcribe(audio: tuple[int, np.ndarray], transcript: str, language: str):
    transcription_response = await groq_client.audio.transcriptions.create(
        file=("audio-file.mp3", audio_to_bytes(audio)),
        model="whisper-large-v3",
        response_format="json",
        language=language,
    )
    transcribed_text = transcription_response.text

    translated_text = "..."

    if transcribed_text and transcribed_text.strip():
        try:
            encoded_text = quote(transcribed_text)
            sl_code = translation_lang.get(language, {}).get('sl_code', 'id')
            url = f"https://ftapi.pythonanywhere.com/translate?sl={sl_code}&dl=id&text={encoded_text}"

            async with httpx.AsyncClient() as client:
                translation_api_response = await client.get(url, timeout=20.0)
                translation_api_response.raise_for_status()
                data = translation_api_response.json()
                translated_text = data.get('destination-text', 'Translation failed.')

        except httpx.RequestError as exc:
            print(f"An error occurred while requesting translation: {exc}")
            translated_text = "Error: Translation request failed."
        except Exception as e:
            print(f"An error occurred during translation processing: {e}")
            translated_text = "Error: Could not parse translation."
    else:
        translated_text = ""

    if transcribed_text or translated_text:
        new_entry = (
            f"--- \n"
            f"Transcription: {transcribed_text}\n"
            f"Translation (ID): {translated_text}"
        )
        yield AdditionalOutputs(transcript + "\n" + new_entry)
    else:
        yield


async def transcribe_audio_file(file: UploadFile, language: str):
    try:
        file_bytes = await file.read()

        transcription_response = await groq_client.audio.transcriptions.create(
            file=(file.filename or "audio.mp3", file_bytes),
            model="whisper-large-v3",
            response_format="json",
            language=language,
        )
        transcribed_text = transcription_response.text
    except Exception as e:
        print(f"Transcription error: {e}")
        raise ValueError("Transcription failed. Please check the audio file format and content.")

    translated_text = "..."

    if transcribed_text and transcribed_text.strip():
        try:
            encoded_text = quote(transcribed_text)
            sl_code = translation_lang.get(language, {}).get('sl_code', 'id')
            url = f"https://ftapi.pythonanywhere.com/translate?sl={sl_code}&dl=id&text={encoded_text}"

            async with httpx.AsyncClient() as client:
                translation_api_response = await client.get(url, timeout=20.0)
                translation_api_response.raise_for_status()
                data = translation_api_response.json()
                translated_text = data.get('destination-text', 'Translation failed.')

        except httpx.RequestError as exc:
            print(f"An error occurred while requesting translation: {exc}")
            translated_text = "Error: Translation request failed."
        except Exception as e:
            print(f"An error occurred during translation processing: {e}")
            translated_text = "Error: Could not parse translation."
    else:
        translated_text = ""

    return {
        "transcript": transcribed_text,
        "translation": translated_text
    }
