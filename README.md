# RAG with Chroma, LangChain, and Google Gemini

A modular Retrieval-Augmented Generation (RAG) project using Chroma vector database, LangChain, and Google Gemini. It supports multiple chunking strategies and retrieval modes, with a simple Streamlit UI.

## Features
- Chroma-only vector store (persistent `vectorstore/`)
- Multiple chunking strategies: fixed, recursive, semantic
- Retrieval modes: similarity, MMR, similarity score threshold
- Modular architecture (`rag/` package)
- Streamlit application for upload and querying

## Architecture
```
rag/
  loaders.py      # File/text extraction
  chunking.py     # Fixed, Recursive, Semantic chunking
  vectorstore.py  # Chroma vector store management
  retrieval.py    # Retriever configs (similarity/MMR/threshold)
  llm.py          # Google Gemini chat & embeddings
  pipeline.py     # Orchestration: process_and_store, ask_question
app.py            # Streamlit UI
utils.py          # Thin wrapper delegating to rag.pipeline
vectorstore/      # Persistent Chroma data
```

## Requirements
- Python 3.10+
- Google Gemini API key (`GOOGLE_API_KEY`)

Activate .venv:
.venv\Scripts\Activate.ps1

Install dependencies:
```bash
pip install -r requirements.txt
```

Set environment variable (PowerShell on Windows):
```powershell
$env:GOOGLE_API_KEY = "YOUR_KEY_HERE"
```

Or on bash (Linux/macOS):
```bash
export GOOGLE_API_KEY="YOUR_KEY_HERE"
```

## Run (Streamlit UI)
```bash
streamlit run app.py
```
Open the URL shown in the terminal (usually `http://localhost:8500`).

## Usage
1. In the sidebar, upload a `.pdf` or `.txt` file.
2. Pick chunking strategy and tune parameters.
3. Choose retrieval type and top-k.
4. Ask your question in the main input and click "Get Answer".

### Chunking strategies
- fixed: splits by characters with fixed window and optional overlap
- recursive: hierarchical split using paragraph/newline/space fallbacks
- semantic: sentence-aware grouping by approximate token budget

### Retrieval modes
- similarity: standard nearest neighbors by embedding similarity
- mmr: Maximal Marginal Relevance (less redundancy)
- similarity_score_threshold: only return chunks above a score threshold

## CLI-free Operation
This project is designed to run via Streamlit. The previous monolithic CLI has been removed in favor of the modular pipeline and UI.

## Data Persistence
- Chroma data persists under `vectorstore/`.
- Re-running with the same directory will reuse the existing collection.

## Troubleshooting
- Missing API key: ensure `GOOGLE_API_KEY` is set in your environment before running.
- No results or poor answers: try a different chunking strategy, increase `chunk_size`, or adjust `k` and retrieval type.
- Large PDFs: prefer `recursive` or `semantic` chunking to maintain context.