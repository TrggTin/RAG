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

---

# RAG với Chroma, LangChain, và Google Gemini

Dự án RAG mô-đun sử dụng CSDL vector Chroma, LangChain, và Google Gemini. Hỗ trợ nhiều phương pháp chia đoạn và kiểu truy hồi, kèm UI Streamlit dễ dùng.

## Tính năng
- Chỉ dùng Chroma (lưu trữ tại `vectorstore/`)
- Nhiều chiến lược chia đoạn: fixed, recursive, semantic
- Kiểu truy hồi: similarity, MMR, similarity score threshold
- Kiến trúc mô-đun (`rag/`)
- Ứng dụng Streamlit để tải tài liệu và hỏi đáp

## Kiến trúc
```
rag/
  loaders.py      # Trích xuất nội dung
  chunking.py     # Chia đoạn Fixed/Recursive/Semantic
  vectorstore.py  # Quản lý Chroma
  retrieval.py    # Cấu hình truy hồi
  llm.py          # Gemini chat & embeddings
  pipeline.py     # Luồng xử lý: process_and_store, ask_question
app.py            # Giao diện Streamlit
utils.py          # Gọi sang rag.pipeline
vectorstore/      # Dữ liệu Chroma
```

## Yêu cầu
- Python 3.10+
- API key Google Gemini (`GOOGLE_API_KEY`)

Cài đặt thư viện:
```bash
pip install -r requirements.txt
```

Thiết lập biến môi trường (PowerShell/Windows):
```powershell
$env:GOOGLE_API_KEY = "YOUR_KEY_HERE"
```

Hoặc trên bash (Linux/macOS):
```bash
export GOOGLE_API_KEY="YOUR_KEY_HERE"
```

## Chạy ứng dụng (Streamlit)
```bash
streamlit run app.py
```
Mở URL hiện trên terminal (thường là `http://localhost:8500`).

## Sử dụng
1. Ở thanh bên, tải tệp `.pdf` hoặc `.txt`.
2. Chọn chiến lược chia đoạn và tinh chỉnh tham số.
3. Chọn kiểu truy hồi và top-k.
4. Nhập câu hỏi và nhấn "Get Answer".

### Chiến lược chia đoạn
- fixed: chia theo ký tự với cửa sổ cố định và overlap
- recursive: chia phân cấp theo đoạn/dòng/khoảng trắng
- semantic: gộp theo câu dựa trên hạn mức token xấp xỉ

### Kiểu truy hồi
- similarity: lân cận gần nhất theo độ tương đồng
- mmr: giảm trùng lặp nội dung
- similarity_score_threshold: chỉ nhận các mảnh có điểm trên ngưỡng

## Lưu trữ dữ liệu
- Dữ liệu Chroma được lưu tại `vectorstore/`.
- Chạy lại sẽ tái sử dụng collection hiện có.

## Khắc phục sự cố
- Thiếu API key: cần đặt `GOOGLE_API_KEY` trước khi chạy.
- Kết quả chưa tốt: thử chiến lược chia đoạn khác, tăng `chunk_size`, hoặc điều chỉnh `k` và kiểu truy hồi.
- PDF lớn: ưu tiên `recursive` hoặc `semantic` để giữ ngữ cảnh. 