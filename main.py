import os
import uuid
import asyncio
import whisper

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

from features import (
    sentiment_analysis,
    summarize_text,
    translate_text,
    extract_entities
)

app = FastAPI(title="Whisper Speech To Text API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = whisper.load_model("base")

SUPPORTED = {"mp3", "wav", "m4a", "mp4", "ogg", "webm"}
MAX_SIZE = 20 * 1024 * 1024


def check_api_key(key: str):
    if not key:
        raise HTTPException(401, "Missing RapidAPI key")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    summarize: bool = Form(False),
    sentiment: bool = Form(False),
    translate_to: str | None = Form(None),
    entities: bool = Form(False),
    timestamps: bool = Form(False),
    x_rapidapi_key: str = Header(None)
):

    check_api_key(x_rapidapi_key)

    ext = file.filename.split(".")[-1].lower()
    if ext not in SUPPORTED:
        raise HTTPException(400, "Unsupported format")

    data = await file.read()

    if len(data) > MAX_SIZE:
        raise HTTPException(413, "File too large")

    temp = f"temp_{uuid.uuid4()}.{ext}"

    with open(temp, "wb") as f:
        f.write(data)

    try:
        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(
            None,
            lambda: model.transcribe(temp, word_timestamps=True)
        )

        text = result["text"]
        lang = result["language"]

        word_ts = None
        if timestamps:
            word_ts = []
            for seg in result["segments"]:
                for w in seg.get("words", []):
                    word_ts.append(w)

        summ = None
        if summarize:
            summ = summarize_text(text)

        senti = None
        if sentiment:
            senti = sentiment_analysis(text)

        trans = None
        if translate_to:
            trans = translate_text(text, translate_to)

        ents = None
        if entities:
            ents = extract_entities(text)

        return {
            "transcript": text,
            "language": lang,
            "summary": summ,
            "sentiment": senti,
            "translation": trans,
            "entities": ents,
            "word_timestamps": word_ts
        }

    finally:
        os.remove(temp)