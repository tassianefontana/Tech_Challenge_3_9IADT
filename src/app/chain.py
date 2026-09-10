"""Pipeline LangChain do assistente medico.

Combina tres fontes de informacao numa unica resposta rastreavel:
  1. LLM customizada (fine-tunada em PubMedQA)
  2. Evidencia cientifica recuperada do banco vetorial (RAG)
  3. Prontuario estruturado do paciente (SQLite)

A resposta sempre carrega as fontes utilizadas (PMIDs e id do paciente),
atendendo ao requisito de explainability.
"""

from typing import Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config import RETRIEVER_TOP_K, SYSTEM_PROMPT
from src.db.patients import format_patient, get_patient

ASSISTANT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\n\n"
     "Regras obrigatorias:\n"
     "1. Baseie-se exclusivamente na evidencia e no prontuario fornecidos.\n"
     "2. Se a informacao for insuficiente, declare isso explicitamente.\n"
     "3. Cite o PMID da evidencia que sustenta cada afirmacao.\n"
     "4. Nunca indique dose ou posologia; descreva a conduta e encaminhe "
     "para validacao medica."),
    ("human",
     "Pergunta do medico:\n{question}\n\n"
     "--- Contexto do paciente ---\n{patient_context}\n\n"
     "--- Evidencia cientifica recuperada ---\n{evidence}\n\n"
     "Responda de forma objetiva, citando os PMIDs utilizados."),
])

NO_PATIENT = "Nenhum paciente informado; responda de forma geral."


def format_evidence(docs: List) -> str:
    """Numera os trechos recuperados e prefixa o PMID para citacao."""
    if not docs:
        return "Nenhuma evidencia relevante encontrada na base."

    blocos = []
    for i, doc in enumerate(docs, 1):
        pmid = doc.metadata.get("pmid", "desconhecido")
        blocos.append(f"[{i}] PMID {pmid}\n{doc.page_content}")
    return "\n\n".join(blocos)


def build_sources(docs: List, patient_id: Optional[str]) -> List[Dict]:
    """Monta a lista de fontes devolvida junto da resposta."""
    sources = [
        {
            "tipo": "evidencia_cientifica",
            "pmid": doc.metadata.get("pmid"),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{doc.metadata.get('pmid')}/",
            "conclusao": doc.metadata.get("label"),
            "trecho": doc.page_content[:300],
        }
        for doc in docs
    ]
    if patient_id:
        sources.append({
            "tipo": "prontuario",
            "paciente_id": patient_id,
            "origem": "data/patients.db",
        })
    return sources


def ask(
    question: str,
    patient_id: Optional[str] = None,
    k: int = RETRIEVER_TOP_K,
) -> Dict:
    """Executa o pipeline completo e devolve resposta + fontes.

    Retorna: {"answer": str, "sources": list, "patient_id": str | None}
    """
    from src.app.llm import get_llm
    from src.rag.vectorstore import get_retriever

    patient_context = NO_PATIENT
    if patient_id:
        patient = get_patient(patient_id)
        if patient is None:
            return {
                "answer": f"Paciente {patient_id} nao encontrado na base.",
                "sources": [],
                "patient_id": patient_id,
            }
        patient_context = format_patient(patient)

    docs = get_retriever(k).invoke(question)

    chain = ASSISTANT_PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke({
        "question": question,
        "patient_context": patient_context,
        "evidence": format_evidence(docs),
    })

    return {
        "answer": answer.strip(),
        "sources": build_sources(docs, patient_id),
        "patient_id": patient_id,
    }
