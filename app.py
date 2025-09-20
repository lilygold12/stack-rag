from fastapi import FastAPI
import sqlite3
import io
from typing import List
from fastapi import UploadFile, File
from fastapi.responses import HTMLResponse
from PyPDF2 import PdfReader
import requests
import os
from sentence_transformers import SentenceTransformer
import numpy as np

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
DB_PATH = "rag.db"
app = FastAPI()
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)


# Connect to database
def get_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys=ON")
    return con

# Create documents table, chunks table, embeddings table
def init_db():
    con = get_db()
    # Documents table
    con.execute("""
        CREATE TABLE IF NOT EXISTS documents(
            id TEXT PRIMARY KEY,
            filename TEXT
        )
    """)
    # Text table
    con.execute("""
        CREATE TABLE IF NOT EXISTS chunks(
            id TEXT PRIMARY KEY,
            doc_id TEXT,
            text TEXT,
            FOREIGN KEY(doc_id) REFERENCES documents(id)
        )
    """)
    # Embeddings table
    con.execute("""
            CREATE TABLE IF NOT EXISTS embeddings(
                chunk_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                dims INTEGER NOT NULL,
                FOREIGN KEY(chunk_id) REFERENCES chunks(id)
            )
        """)
    con.commit()
    con.close()


# Turn strings into vectors
def embed_texts(texts: list[str]) -> np.ndarray:
    vecs = embed_model.encode(texts, normalize_embeddings=True)
    return vecs.astype(np.float32)


# Help store embeddings
def store_embeddings(con, chunk_ids: list[str], vectors: np.ndarray):
    dims = vectors.shape[1]
    rows = [(cid, vectors[i].tobytes(), int(dims)) for i, cid in enumerate(chunk_ids)]
    con.executemany(
        "INSERT OR REPLACE INTO embeddings(chunk_id, vector, dims) VALUES (?, ?, ?)",
        rows
    )
    con.commit()


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


# Help find places in the text that have most overlap with query
def overlap_score(q: str, text: str) -> float:
    q_tokens = set(tokenize(q))
    t_tokens = set(tokenize(text))
    if not q_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


# Words in query to ignore
CUT = {"a","an","and","are","for","from","i","if","is","it","its","of","or","that","the","there","this","to","with"}


# Help tokenize the query
def tokenize(s: str):
    out, w = [], []
    for ch in s.lower():
        if ch.isalnum():
            w.append(ch)
        else:
            if w:
                tok = "".join(w)
                if tok not in CUT:
                    out.append(tok)
                w = []
    if w:
        tok = "".join(w)
        if tok not in CUT:
            out.append(tok)
    return out

# Return chunks most similar in meaning
def semantic_search(query: str, top_k: int = 5):
    # Turn query into vector
    q_vec = embed_texts([query])[0]

    con = get_db()
    cur = con.cursor()
    # Get chunk embedding and original text
    cur.execute("""
        SELECT e.vector, e.dims, c.doc_id, c.id, c.text
        FROM embeddings e
        JOIN chunks c ON c.id = e.chunk_id
    """)
    rows = cur.fetchall()
    con.close()

    if not rows:
        return []

    # Get similarity score for each chunk
    scored = []
    for blob, dims, doc_id, chunk_id, text in rows:
        v = np.frombuffer(blob, dtype=np.float32, count=dims)
        score = float(np.dot(q_vec, v))
        scored.append((score, doc_id, chunk_id, text))

    # Sort by similarity score, keep top_k most relevant
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# Hybrid search to combine both semantic and keyword results
def hybrid_search(q: str, top_k: int = 3, alpha: float = 0.7):
    # Get top results from semantic search
    semantic_results = semantic_search(q, top_k=20)
    print(f"[hybrid] semantic_results={len(semantic_results)}")

    scored = []
    for sem_score, doc_id, chunk_id, text in semantic_results:
        keyword_score = overlap_score(q, text)
        final_score = alpha * sem_score + (1 - alpha) * keyword_score
        scored.append((final_score, doc_id, chunk_id, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# When server starts run init_db
@app.on_event("startup")
def startup_event():
    init_db()


@app.get("/", response_class=HTMLResponse)
def home():
    with open("RAG-UI.html", "r", encoding="utf-8") as f:
        return f.read()


# Ingest endpoint
@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    con = get_db()
    cur = con.cursor()
    results = []

    # Read file, extract text, and split into chunks
    for file in files:
        data = await file.read()
        reader = PdfReader(io.BytesIO(data))
        text = " ".join([page.extract_text() or "" for page in reader.pages])
        chunks = text_chunks(text)

        # Save documents and chunks into SQLite
        doc_id = file.filename  # use filename as id
        cur.execute("INSERT INTO documents (id, filename) VALUES (?, ?)",
                    (doc_id, file.filename))

        new_chunk_ids: list[str] = []
        new_chunk_texts: list[str] = []

        # Save chunk rows
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"
            cur.execute("INSERT INTO chunks (id, doc_id, text) VALUES (?, ?, ?)",
                        (chunk_id, doc_id, chunk))
            new_chunk_ids.append(chunk_id)
            new_chunk_texts.append(chunk)

        # Embed and store vectors
        if new_chunk_texts:
            vecs = embed_texts(new_chunk_texts)
            store_embeddings(con, new_chunk_ids, vecs)

        results.append({"filename": file.filename, "num_chunks": len(chunks)})

    con.commit()
    con.close()

    return {"ingested": results}

# Query endpoint
@app.post("/query")
def query(q: str, top_k: int = 3, alpha: float = 0.7):
    # Detect if knowledge search is necessary
    if q.lower() in {"hi", "hello", "thanks", "thank you", "ok"}:
        return {"answer": "Hello! Upload your PDFs, then ask a question."}

    # Run hybrid search to find chunks with most relevant meaning and keyword overlap
    results = hybrid_search(q, top_k=top_k, alpha=alpha)
    if not results:
        return {"answer": "Please upload a PDF."}

    # Confidence threshold
    best_score = results[0][0]
    if best_score < 0.2:
        return {"answer": "Insufficient evidence."}

    # Add the chunk text to the context
    context = "\n\n".join([r[3] for r in results])

    # Add LLM prompt and call Mistral API
    if not MISTRAL_API_KEY:
        return {"error": "Missing MISTRAL_API_KEY"}

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


