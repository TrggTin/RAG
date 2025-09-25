from typing import Any


def get_retriever(store, search_type: str = "similarity", k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.5) -> Any:
    s = search_type.lower()
    if s == "mmr":
        return store.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult})
    if s in ("similarity_score_threshold", "threshold"):
        return store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.2, "k": k})
    return store.as_retriever(search_type="similarity", search_kwargs={"k": k}) 