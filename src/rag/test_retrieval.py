"""Teste rapido de recuperacao no banco vetorial.

Uso:
    python -m src.rag.test_retrieval "sua pergunta clinica"
"""

import sys

from src.rag.vectorstore import get_retriever

DEFAULT_QUERY = "Do mitochondria play a role in programmed cell death?"


def run(query: str) -> None:
    docs = get_retriever().invoke(query)

    if not docs:
        print("[erro] nenhum documento retornado. Rode: "
              "python -m src.rag.build_vectordb")
        return

    print(f"Consulta: {query}\n")
    for i, doc in enumerate(docs, 1):
        print("=" * 70)
        print(f"[{i}] fonte: {doc.metadata.get('source')} "
              f"| label: {doc.metadata.get('label')}")
        print(doc.page_content[:600])


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY)
