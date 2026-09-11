"""Pipeline LangChain do assistente medico.

Combina tres fontes de informacao numa unica resposta rastreavel:
  1. LLM customizada (fine-tunada em PubMedQA)
  2. Evidencia cientifica recuperada do banco vetorial (RAG)
  3. Prontuario estruturado do paciente (SQLite)

A resposta sempre carrega as fontes utilizadas (PMIDs e id do paciente),
atendendo ao requisito de explainability.

Todas as etapas passam pelos guardrails de `src/security/` e sao gravadas
na trilha de auditoria sob um mesmo `trace_id`.
"""

from typing import Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config import RETRIEVER_TOP_K, SYSTEM_PROMPT
from src.db.patients import format_patient, get_patient
from src.security import audit, guardrails

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
    trace_id: Optional[str] = None,
) -> Dict:
    """Executa o pipeline completo e devolve resposta + fontes.

    Retorna:
        {"answer", "sources", "patient_id", "trace_id", "guardrail"}
    """
    from src.app.llm import get_llm
    from src.rag.vectorstore import get_retriever

    trace_id = trace_id or audit.new_trace_id()

    # --- guardrail de entrada ---------------------------------------
    gate = guardrails.check_input(question)
    audit.log_event(
        audit.EVENT_INPUT,
        trace_id,
        {"pergunta": gate.text, "violacoes": gate.violations},
        patient_id=patient_id,
    )
    if not gate.allowed:
        return {
            "answer": gate.text,
            "sources": [],
            "patient_id": patient_id,
            "trace_id": trace_id,
            "guardrail": gate,
        }
    question = gate.text

    # --- contexto do paciente ---------------------------------------
    patient_context = NO_PATIENT
    if patient_id:
        patient = get_patient(patient_id)
        if patient is None:
            audit.log_event(
                audit.EVENT_ERROR, trace_id,
                {"motivo": "paciente_inexistente"}, patient_id=patient_id,
            )
            return {
                "answer": f"Paciente {patient_id} nao encontrado na base.",
                "sources": [],
                "patient_id": patient_id,
                "trace_id": trace_id,
                "guardrail": None,
            }
        patient_context = format_patient(patient)
        audit.log_event(
            audit.EVENT_PATIENT, trace_id,
            {"diagnostico": patient["diagnostico"],
             "exames_pendentes": sum(
                 1 for e in patient["exames"] if e["status"] == "pendente")},
            patient_id=patient_id,
        )

    # --- recuperacao de evidencia -----------------------------------
    docs = get_retriever(k).invoke(question)
    audit.log_event(
        audit.EVENT_RETRIEVAL,
        trace_id,
        {"k": k, "pmids": [d.metadata.get("pmid") for d in docs]},
        patient_id=patient_id,
    )

    # --- geracao -----------------------------------------------------
    chain = ASSISTANT_PROMPT | get_llm() | StrOutputParser()
    raw_answer = chain.invoke({
        "question": question,
        "patient_context": patient_context,
        "evidence": format_evidence(docs),
    })
    audit.log_event(
        audit.EVENT_LLM, trace_id,
        {"caracteres": len(raw_answer)}, patient_id=patient_id,
    )

    # --- guardrail de saida ------------------------------------------
    checked = guardrails.validate_answer(
        raw_answer.strip(),
        has_evidence=bool(docs),
        trace_id=trace_id,
        patient_id=patient_id,
    )

    return {
        "answer": checked.text,
        "sources": build_sources(docs, patient_id),
        "patient_id": patient_id,
        "trace_id": trace_id,
        "guardrail": checked,
    }
