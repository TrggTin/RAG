import os
from pathlib import Path
from typing import List, Optional
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)


class VectorStoreManager:
    def __init__(self, backend: str = "chroma", persist_directory: str = "vectorstore", collection_name: str = "rag_chunks", embeddings=None):
        self.backend = backend.lower()
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = embeddings or get_embeddings()
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.store = None

    def create_from_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        if self.backend == "chroma":
            self.store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            self.store.add_texts(texts, metadatas=metadatas)
            self.store.persist()
        else:
            self.store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
            self.save()
        return self.store

    def save(self):
        if self.backend == "chroma":
            if self.store:
                self.store.persist()
        else:
            index_path = Path(self.persist_directory) / "faiss_index"
            self.store.save_local(str(index_path))

    def load(self) -> bool:
        try:
            if self.backend == "chroma":
                self.store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory,
                )
                return True
            index_path = Path(self.persist_directory) / "faiss_index"
            if not index_path.exists():
                return False
            self.store = FAISS.load_local(str(index_path), self.embeddings, allow_dangerous_deserialization=True)
            return True
        except Exception:
            return False

    def as_retriever(self, search_type: str = "similarity", k: int = 5):
        return self.store.as_retriever(search_type=search_type, search_kwargs={"k": k}) 