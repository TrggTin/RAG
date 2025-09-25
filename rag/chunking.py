from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from sentence_transformers import SentenceTransformer


class ChunkingStrategy:
    def split(self, text: str) -> List[str]:
        raise NotImplementedError


class FixedChunking(ChunkingStrategy):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        splitter = CharacterTextSplitter(
            separator=" ", chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        return splitter.split_text(text)


class RecursiveChunking(ChunkingStrategy):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text)


class SemanticChunking(ChunkingStrategy):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_chunk_tokens: int = 200):
        self.model = SentenceTransformer(model_name)
        self.max_chunk_tokens = max_chunk_tokens

    def split(self, text: str) -> List[str]:
        # Simple semantic segmentation: split by sentences, then greedily merge until token budget
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' ').strip())
        chunks: List[str] = []
        current: List[str] = []

        def approx_tokens(s: str) -> int:
            # rough approximation
            return max(1, len(s.split()) // 0.75)

        for sent in sentences:
            candidate = (" ".join(current + [sent])).strip()
            if not candidate:
                continue
            if approx_tokens(candidate) <= self.max_chunk_tokens:
                current.append(sent)
            else:
                if current:
                    chunks.append(" ".join(current).strip())
                current = [sent]
        if current:
            chunks.append(" ".join(current).strip())
        return [c for c in chunks if c]


def get_chunker(strategy: str, **kwargs) -> ChunkingStrategy:
    s = strategy.lower()
    if s == "fixed":
        return FixedChunking(**kwargs)
    if s == "recursive":
        return RecursiveChunking(**kwargs)
    if s == "semantic":
        return SemanticChunking(**kwargs)
    return RecursiveChunking(**kwargs) 