from fastapi import FastAPI
import sqlite3
import time
from typing import List
import hashlib
import io
from typing import List
from fastapi import UploadFile, File
from PyPDF2 import PdfReader


# Break long text into 500 word chunks to be read by LLM
def text_chunks(text: str, words_per_chunk: int = 500):
    words = text.split()
    out, i = [], 0
    while i < len(words):
        part = words[i:i+words_per_chunk]
        if not part:
            break
        out.append(" ".join(part))
        i += words_per_chunk
    return out


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "FastAPI test"}

# ingest endpoint
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    data = await file.read()
    reader = PdfReader(io.BytesIO(data))
    text = " ".join([page.extract_text() or "" for page in reader.pages])
    return {"filename": file.filename, "text": text[:500]}