import streamlit as st
from utils import process_and_store, ask_question

st.set_page_config(page_title="RAG with ChromaDB, LangChain, Gemini", layout="wide")
st.title("RAG with ChromaDB, LangChain, and Google Gemini")

st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF or TXT", type=["pdf", "txt"])

# Chunking options
st.sidebar.subheader("Chunking Settings")
chunking_strategy = st.sidebar.selectbox("Chunking Strategy", ["recursive", "fixed", "semantic"], index=0)
chunk_size = st.sidebar.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
chunk_overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=50)
semantic_max_tokens = st.sidebar.number_input("Semantic max tokens", min_value=50, max_value=1000, value=200, step=50)

# Retrieval options
st.sidebar.subheader("Retrieval Settings")
search_type = st.sidebar.selectbox("Search type", ["similarity", "mmr", "similarity_score_threshold"], index=0)
k = st.sidebar.slider("Top-k", min_value=1, max_value=20, value=5)

if uploaded_file:
    with st.spinner("Processing document..."):
        process_and_store(
            uploaded_file,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            semantic_max_tokens=semantic_max_tokens,
            vector_backend="chroma",
        )
    st.sidebar.success("Document processed and stored!")

st.header("Ask a Question")
question = st.text_input("Enter your question about the uploaded document:")

if st.button("Get Answer") and question:
    with st.spinner("Retrieving and generating answer..."):
        try:
            answer, retrieved = ask_question(
                question,
                search_type=search_type,
                k=k,
                vector_backend="chroma",
            )
            st.subheader("Answer")
            st.write(answer)
            st.subheader("Top Retrieved Chunks")
            for i, chunk in enumerate(retrieved):
                st.markdown(f"**Chunk {i+1}:** {chunk}")
        except Exception as e:
            st.error(f"Error: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Set your Google Gemini API key as the environment variable `GOOGLE_API_KEY` before running.")
