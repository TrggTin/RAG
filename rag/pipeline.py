from typing import List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .loaders import extract_text
from .chunking import get_chunker
from .vectorstore import VectorStoreManager, get_embeddings
from .retrieval import get_retriever
from .llm import get_gemini_chat


def process_and_store(file,
                      chunking_strategy: str = "recursive",
                      chunk_size: int = 1000,
                      chunk_overlap: int = 200,
                      semantic_max_tokens: int = 200,
                      vector_backend: str = "chroma",
                      embeddings_model: str = "all-MiniLM-L6-v2",
                      persist_dir: str = "vectorstore",
                      collection_name: str = "rag_chunks") -> None:
    text = extract_text(file)
    chunker = get_chunker(
        chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_chunk_tokens=semantic_max_tokens
    )
    chunks = chunker.split(text)

    embeddings = get_embeddings(embeddings_model)
    vsm = VectorStoreManager(backend=vector_backend, persist_directory=persist_dir, collection_name=collection_name, embeddings=embeddings)
    metadatas = [{"source": getattr(file, "name", "uploaded")} for _ in chunks]
    vsm.create_from_texts(chunks, metadatas=metadatas)


def build_rag_chain(search_type: str = "similarity", k: int = 5,
                    vector_backend: str = "chroma",
                    embeddings_model: str = "all-MiniLM-L6-v2",
                    persist_dir: str = "vectorstore",
                    collection_name: str = "rag_chunks"):
    embeddings = get_embeddings(embeddings_model)
    vsm = VectorStoreManager(backend=vector_backend, persist_directory=persist_dir, collection_name=collection_name, embeddings=embeddings)
    if not vsm.load():
        raise ValueError("No vectorstore found. Please process a document first.")

    retriever = get_retriever(vsm.store, search_type=search_type, k=k)
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the question based on the context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
    )
    llm = get_gemini_chat()

    def format_docs(docs: List) -> str:
        return "\n\n---\n\n".join([d.page_content for d in docs])

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask_question(question: str,
                 search_type: str = "similarity",
                 k: int = 5,
                 vector_backend: str = "chroma",
                 embeddings_model: str = "all-MiniLM-L6-v2",
                 persist_dir: str = "vectorstore",
                 collection_name: str = "rag_chunks") -> Tuple[str, List[str]]:
    chain = build_rag_chain(search_type, k, vector_backend, embeddings_model, persist_dir, collection_name)
    answer = chain.invoke(question)

    # Also return top chunks for UI
    embeddings = get_embeddings(embeddings_model)
    vsm = VectorStoreManager(backend=vector_backend, persist_directory=persist_dir, collection_name=collection_name, embeddings=embeddings)
    vsm.load()
    retriever = get_retriever(vsm.store, search_type=search_type, k=k)
    docs = retriever.get_relevant_documents(question)
    retrieved = [d.page_content for d in docs]
    return answer, retrieved 