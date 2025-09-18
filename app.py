from fastapi import FastAPI
import sqlite3
import io
from typing import List
from fastapi import UploadFile, File
from PyPDF2 import PdfReader
import requests
import os

MISTRAL_API_KEY = "kPg3un3taQ1yTpIn2RDtWYnOMDFpooBP"
DB_PATH = "rag.db"
app = FastAPI()


# Connect to database
def get_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys=ON")
    return con

# Create documents table and chunks table
def init_db():
    con = get_db()
    con.execute("""
        CREATE TABLE IF NOT EXISTS documents(
            id TEXT PRIMARY KEY,
            filename TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS chunks(
            id TEXT PRIMARY KEY,
            doc_id TEXT,
            text TEXT,
            FOREIGN KEY(doc_id) REFERENCES documents(id)
        )
    """)
    con.commit()
    con.close()


# When server starts run init_db
@app.on_event("startup")
def startup_event():
    init_db()


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


@app.get("/")
def read_root():
    return {"message": "FastAPI test"}


# ingest endpoint
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    # Read file, extract text, and split into chunks
    data = await file.read()
    reader = PdfReader(io.BytesIO(data))
    text = " ".join([page.extract_text() or "" for page in reader.pages])
    chunks = text_chunks(text)

    # Save documents and chunks into SQLite
    con = get_db()
    cur = con.cursor()
    doc_id = file.filename  # use filename as id
    cur.execute("INSERT INTO documents (id, filename) VALUES (?, ?)",
                (doc_id, file.filename))

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_{i}"
        cur.execute("INSERT INTO chunks (id, doc_id, text) VALUES (?, ?, ?)",
                    (chunk_id, doc_id, chunk))

    con.commit()
    con.close()

    return {"filename": file.filename, "num_chunks": len(chunks)}

# Query endpoint
@app.post("/query")
def query(q: str):
    con = get_db()
    cur = con.cursor()

    # Find top 3 chunks that contain the query text
    cur.execute("SELECT doc_id, text FROM chunks WHERE text LIKE ? LIMIT 3", (f"%{q}%",))
    rows = cur.fetchall()
    con.close()

    if not rows:
        return {"answer": "No matching text found."}


    context = "\n\n".join([r[0] for r in rows])

    # Add LLM prompt and call Mistral API
    prompt = f"Answer the question using only the text below. If not enough information say 'Insufficient evidence'.\n\nContext:\n{context}\n\nQuestion: {q}"

    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
        json={
            "model": "mistral-small",
            "messages": [{"role": "user", "content": prompt}]
        }
    )

    if resp.status_code >= 300:
        return {"mistral_status": resp.status_code, "mistral_body": resp.text}

    data = resp.json()
    answer = data["choices"][0]["message"]["content"]

    # Return the query and the answer
    return {"query": q, "answer": answer}