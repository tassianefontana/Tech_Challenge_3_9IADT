"""Acesso unico ao banco vetorial e ao modelo de embeddings."""

from functools import lru_cache

from src.config import (
    EMBEDDING_MODEL,
    RETRIEVER_TOP_K,
    VECTORDB_COLLECTION,
    VECTORDB_DIR,
)


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_vectorstore():
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=VECTORDB_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(VECTORDB_DIR),
    )


def get_retriever(k: int = RETRIEVER_TOP_K):
    return get_vectorstore().as_retriever(search_kwargs={"k": k})
