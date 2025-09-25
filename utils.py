import uuid
from typing import Tuple, List

from rag.pipeline import process_and_store as _process_and_store
from rag.pipeline import ask_question as _ask_question


def process_and_store(file,
                      chunking_strategy: str = "recursive",
                      chunk_size: int = 1000,
                      chunk_overlap: int = 200,
                      semantic_max_tokens: int = 200,
                      vector_backend: str = "chroma",
                      embeddings_model: str = "all-MiniLM-L6-v2",
                      persist_dir: str = "vectorstore",
                      collection_name: str = "rag_chunks") -> None:
    _process_and_store(
        file,
        chunking_strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        semantic_max_tokens=semantic_max_tokens,
        vector_backend=vector_backend,
        embeddings_model=embeddings_model,
        persist_dir=persist_dir,
        collection_name=collection_name,
    )


def ask_question(question: str,
                 search_type: str = "similarity",
                 k: int = 5,
                 vector_backend: str = "chroma",
                 embeddings_model: str = "all-MiniLM-L6-v2",
                 persist_dir: str = "vectorstore",
                 collection_name: str = "rag_chunks") -> Tuple[str, List[str]]:
    return _ask_question(
        question,
        search_type=search_type,
        k=k,
        vector_backend=vector_backend,
        embeddings_model=embeddings_model,
        persist_dir=persist_dir,
        collection_name=collection_name,
    )
