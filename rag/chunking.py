from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

class ChunkingStrategy:
    def split(self, text: str) -> List[str]:
        raise NotImplementedError

class FixedChunking(ChunkingStrategy):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        if not text.strip():
            return []
        splitter = CharacterTextSplitter(
            separator=" ", 
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_text(text)

class RecursiveChunking(ChunkingStrategy):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        if not text.strip():
            return []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text)

class SemanticChunking(ChunkingStrategy):
    def __init__(self, max_chunk_tokens: int = 200):
        self.max_chunk_tokens = max_chunk_tokens

    def split(self, text: str) -> List[str]:
        if not text.strip():
            return []
            
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' ').strip())
        if not sentences:
            return []
            
        chunks = []
        current = []

        def approx_tokens(s: str) -> int:
            return max(1, int(len(s.split()) / 0.75))

        for sent in sentences:
            if not sent.strip():
                continue
                
            candidate = (" ".join(current + [sent])).strip()
            if approx_tokens(candidate) <= self.max_chunk_tokens:
                current.append(sent)
            else:
                if current:
                    chunks.append(" ".join(current).strip())
                current = [sent]
                
        if current:
            chunks.append(" ".join(current).strip())
            
        return [c for c in chunks if c.strip()]

def get_chunker(strategy: str, **kwargs) -> ChunkingStrategy:
    s = strategy.lower()
    
    if s == "fixed":
        return FixedChunking(
            chunk_size=kwargs.get('chunk_size', 1000),
            chunk_overlap=kwargs.get('chunk_overlap', 0)
        )
    elif s == "recursive":
        return RecursiveChunking(
            chunk_size=kwargs.get('chunk_size', 1000),
            chunk_overlap=kwargs.get('chunk_overlap', 200)
        )
    elif s == "semantic":
        return SemanticChunking(
            max_chunk_tokens=kwargs.get('max_chunk_tokens', 200)
        )
    else:
        return RecursiveChunking(
            chunk_size=kwargs.get('chunk_size', 1000),
            chunk_overlap=kwargs.get('chunk_overlap', 200)
        )
