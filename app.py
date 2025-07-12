import streamlit as st
from utils import process_and_store, ask_question

st.set_page_config(page_title="RAG with ChromaDB, LangChain, Gemini", layout="wide")
st.title("RAG with ChromaDB, LangChain, and Google Gemini")

st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF or TXT", type=["pdf", "txt"])
if uploaded_file:
    with st.spinner("Processing document..."):
        process_and_store(uploaded_file)
    st.sidebar.success("Document processed and stored!")

st.header("Ask a Question")
question = st.text_input("Enter your question about the uploaded document:")

if st.button("Get Answer") and question:
    with st.spinner("Retrieving and generating answer..."):
        try:
            answer, retrieved, scores = ask_question(question)
            st.subheader("Answer")
            st.write(answer)
            st.subheader("Top Retrieved Chunks")
            for i, chunk in enumerate(retrieved):
                st.markdown(f"**Chunk {i+1}:** {chunk}")
            st.subheader("Retrieval Confidence")
        except Exception as e:
            st.error(f"Error: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Set your Google Gemini API key as the environment variable `GOOGLE_API_KEY` before running.")
