from dotenv import load_dotenv
load_dotenv()

from fastapi.middleware.cors import CORSMiddleware
import json
import uuid
from pathlib import Path
from AI_STT_Translation.transcribe import transcribe, get_credentials, transcribe_audio_file
import os
import gradio as gr
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from fastapi.responses import HTMLResponse, StreamingResponse
from fastrtc import (
    ReplyOnPause,
    Stream,
    get_cloudflare_turn_credentials_async,
)
from gradio.utils import get_space
from pydantic import BaseModel


cur_dir = Path(__file__).parent

transcript = gr.Textbox(label="Transcript", lines=10)
language_input = gr.Dropdown(
    label="Source Language",
    choices=['en', 'jv', 'su', 'id', 'cn'],
    value='id'
)
HF_TOKEN = os.getenv("HF_TOKEN")

stream = Stream(
    ReplyOnPause(transcribe),
    modality="audio",
    mode="send",
    additional_inputs=[transcript, language_input],
    additional_outputs=[transcript],
    additional_outputs_handler=lambda a, b: b,
    concurrency_limit=5 if get_space() else None,
    time_limit=90 if get_space() else None,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify ["https://your-frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],  # allow POST, OPTIONS, GET, etc.
    allow_headers=["*"],
)

stream.mount(app)


@app.on_event("startup")
async def init_rtc():
    rtc_config = await get_cloudflare_turn_credentials_async(hf_token=HF_TOKEN)
    stream.rtc_configuration = rtc_config

class SendInput(BaseModel):
    webrtc_id: str
    transcript: str
    language: str


@app.post("/send_input")
def send_input(body: SendInput):
    stream.set_input(body.webrtc_id, body.transcript, body.language)

@app.get("/transcript")
def _(webrtc_id: str):
    async def output_stream():
        async for output in stream.output_stream(webrtc_id):
            last_entry = "\n".join(output.args[0].split("---")[-1:])
            yield f"event: output\ndata: {json.dumps(last_entry)}\n\n"

    return StreamingResponse(output_stream(), media_type="text/event-stream")

@app.get("/")
async def index():
    rtc_config = await get_credentials(huggingface_token=HF_TOKEN)
    webrtc_id = str(uuid.uuid4())
    return JSONResponse(content={
        "webrtc_id": webrtc_id,
        "rtc_config": rtc_config
    })


@app.get("/ai-definition")
async def ai_definition(word: str, lang_src: str = None, lang_dst: str = "id"):
    from RAG_powered_search.rag_powered_search import llm_answer
    
    result = llm_answer(word, lang_src, lang_dst)
    return {"result": result}

@app.get("/ai-search")
async def ai_search(query: str, lang_dst: str = None, lang_src: str = "id"):
    from RAG_powered_search.rag_powered_search import llm_search_from_milvus
    
    results = await llm_search_from_milvus(query, lang_dst, lang_src)
    return {"results": results}

@app.get("/transcribe-audio-file")
async def transcribe_audio(
    file: UploadFile = File(...),
    language_src: str = Form("id"),
    language_dst: str = Form("en")
):
    try:
        result = await transcribe_audio_file(file, language_src, language_dst)
        return JSONResponse(content={"result": result}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import os

    if (mode := os.getenv("MODE")) == "UI":
        stream.ui.launch(server_name="0.0.0.0", server_port=7860)
    elif mode == "PHONE":
        stream.fastphone(host="0.0.0.0", port=7860)
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=7860)