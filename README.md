Simple RAG Pipeline

This project is a Retrieval-Augmented Generation (RAG) system built with Python, FastAPI, and the Mistral API. It ingests PDF files, stores their text and embeddings in a SQLite database, and answers questions by retrieving the most relevant chunks and passing them to a language model.
System Design

1. Data Ingestion
Endpoint: POST /ingest
Upload one or more PDF files at a time. I used PyPDF2 to extract the text.
Chunking is word-based (~500 words each).
Stored in SQLite across three tables: documents, chunks, and embeddings.
Considerations:
I chose 500 words because it balances context size with efficiency.

2. Query Processing
Endpoint: POST /query
Intent check: greetings like “hi” or “thanks” skip the knowledge search.
Tokenization removes stopwords before overlap scoring.
Considerations:
Lightweight intent detection to avoid unnecessary LLM calls.
Filtering out common words to prevent overlap scores from being inflated.

3. Semantic Search
Embeddings created with sentence-transformers/all-MiniLM-L6-v2.
Query embedding is compared with stored vectors using dot product similarity.
Considerations:
Embeddings are normalized so dot products measure cosine similarity.

4. Hybrid Search (Post-processing)
Combines semantic score with keyword overlap.
Weighted average: alpha controls balance (default 0.7). I set alpha = 0.7 so the system relies mainly on semantic similarity, but still respects keyword overlap.
Results re-ranked, then top-k selected. I set top_k = 3 to balance completeness and efficiency. One chunk might miss context, but more than three risks adding irrelevant information.
Considerations:
Pure semantic search can drift, so keywords keep answers grounded.
alpha can be tuned depending on the quality of documents.

5. Generation
Prompt template instructs the model to only use provided text.
Calls the Mistral API.
If top similarity score <0.2, system refuses with “Insufficient evidence”.
Considerations:
Threshold prevents hallucinations when no relevant context exists.

6. UI
Basic HTML form (RAG-UI.html).
Lets you type in a question and see the LLM’s answer.
Considerations:
Focused on backend logic — UI is intentionally minimal.

Diagram of Pipeline:
PDF upload --> chunk --> embed --> store in SQLite
                                       |
                          query --> hybrid search --> top chunks
                                       |
                                prompt --> Mistral API --> answer

How to Run:
1) Install dependencies from terminal
pip install -r requirements.txt
2) Set your Mistral API key
export MISTRAL_API_KEY=your_key_here
3) Start the server
uvicorn app:app --reload
You should see Uvicorn say it’s running at http://127.0.0.1:8000.
On first run, the app will create rag.db automatically.
4) Use the UI
Open: http://127.0.0.1:8000/
Upload PDFs using the form on the page.
Ask a question in the text box and submit.

Endpoints:
POST /ingest → upload one or more PDFs
POST /query → ask a question

Libraries Used:
FastAPI (API framework);
SQLite3 (local DB);
PyPDF2 (PDF parsing);
SentenceTransformers (all-MiniLM-L6-v2);
NumPy (vector math);
Mistral API (LLM calls);

Limitations & Future Work:
If I had more time, I would
Add citations in answers;
Improve chunking (paragraph boundaries instead of word count);
Add post-hoc evidence checks.

Project Checklist:
Data Ingestion:
 ✔ POST /ingest endpoint for multiple PDFs
 ✔ Extracts text with PyPDF2, chunks into ~500 words
 ✔ Stored in SQLite tables for documents, chunks, embeddings
 
Query Processing:
 ✔ /query endpoint
 ✔ Intent detection for greetings
 ✔ Tokenization + stopword removal
 
Semantic Search:
 ✔ SentenceTransformer embeddings
 ✔ Dot product similarity
 
Post-processing:
 ✔ Hybrid search (semantic + keyword overlap)
 ✔ Weighted scoring with adjustable alpha
 ✔ Re-ranking of results
 
Generation:
 ✔ Prompt template enforcing context use
 ✔ Mistral API call
 ✔ Refusal if score < 0.2 (“Insufficient evidence”)
UI:
 ✔ Minimal HTML form (RAG-UI.html)
Extra considerations:
 ✔ No external search library
 ✔ No vector database used (SQLite only)
 ✔ Simple refusal policy via confidence threshold
