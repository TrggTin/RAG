import os
import tempfile
import uuid
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
import PyPDF2

# --- Document Extraction ---
def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        return file.read().decode("utf-8")

# --- Chunking ---
def text_chunk(text, max_length=1000):
    import re
    from collections import deque
    sentences = deque(re.split(r'(?<=[.!?])\s+', text.replace('\n', ' ')))
    chunks = []
    chunk_text = ""
    while sentences:
        sentence = sentences.popleft().strip()
        if len(chunk_text) + len(sentence) > max_length and chunk_text:
            chunks.append(chunk_text)
            chunk_text = sentence
        else:
            chunk_text += " " + sentence
    if chunk_text:
        chunks.append(chunk_text)
    return chunks

# --- Embedding Function ---
def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- ChromaDB Setup --- 
def get_chroma_collection():
    persist_dir = "vectorstore"
    os.makedirs(persist_dir, exist_ok=True)
    embedding_func = get_embedding_function()
    db = Chroma(
        collection_name="rag_chunks",
        embedding_function=embedding_func,
        persist_directory=persist_dir
    )
    return db

# --- Process and Store Document ---
def process_and_store(file):
    text = extract_text(file)
    chunks = text_chunk(text)
    db = get_chroma_collection()
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": file.name}] * len(chunks)
    db.add_texts(chunks, metadatas=metadatas, ids=ids)
    db.persist()

# --- Google Gemini LLM Setup ---
def get_gemini_llm():
    import os
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY environment variable.")
    return GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

# --- Ask Question ---
def ask_question(question, k=5):
    db = get_chroma_collection()
    retriever = db.as_retriever(search_kwargs={"k": k})
    llm = get_gemini_llm()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    result = qa({"query": question})
    answer = result["result"]
    docs = result["source_documents"]
    retrieved = [doc.page_content for doc in docs]
    scores = [doc.metadata.get("score", 1.0) for doc in docs]  # Chroma may not return scores, so default to 1.0
    return answer, retrieved, scores
