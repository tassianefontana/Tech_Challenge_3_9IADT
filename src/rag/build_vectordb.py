import json

from langchain_core.documents import Document

from src.config import PROCESSED_FILE, VECTORDB_COLLECTION, VECTORDB_DIR
from src.rag.vectorstore import get_embeddings


def load_documents() -> list:
    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        text = (
            f"Pergunta: {item['question']}\n\n"
            f"Evidencia:\n{item['context']}\n\n"
            f"Conclusao ({item['label']}): {item['answer']}"
        )
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "pmid": item["id"],
                    "label": item["label"],
                    "source": f"PubMed PMID {item['id']}",
                    "question": item["question"],
                },
            )
        )
    return documents


def run() -> None:
    from langchain_chroma import Chroma

    documents = load_documents()
    print(f"[info] {len(documents)} documentos carregados.")

    Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        collection_name=VECTORDB_COLLECTION,
        persist_directory=str(VECTORDB_DIR),
    )
    print(f"[ok] banco vetorial criado em {VECTORDB_DIR}")


if __name__ == "__main__":
    run()
