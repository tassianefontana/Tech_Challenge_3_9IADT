"""Fluxo de decisao automatizado do assistente medico (LangGraph).

Requisito do Tech Challenge Fase 3: "organizar fluxos de decisao
automatizados e seguros, onde, ao receber informacoes sobre um paciente, o
sistema possa acionar diferentes etapas, como verificar exames pendentes,
sugerir tratamentos e emitir alertas para a equipe medica".

Enquanto `src/app/chain.py` responde uma pergunta em linha reta, aqui o
atendimento vira uma maquina de estados com desvios condicionais,
interrupcao precoce e alertas. Topologia:

    triagem ─┬─(bloqueado)──────────────────────────────────► END
             └─(ok)─► prontuario ─► verificar_exames ─┬─(pendentes)─► alerta_exames ─┐
                                                      └─(nenhum)────────────────────►┴─► buscar_evidencia
    buscar_evidencia ─► sugerir_conduta ─► guardrail ─► alertar_equipe ─► END

Cada no grava seu proprio evento na trilha de auditoria sob o mesmo
`trace_id`, permitindo reconstruir o caminho percorrido.
"""

from typing import Dict, List, Optional

from typing_extensions import TypedDict

from src.app.chain import (
    ASSISTANT_PROMPT,
    NO_PATIENT,
    build_sources,
    format_evidence,
)
from src.config import RETRIEVER_TOP_K
from src.db.patients import format_patient, get_patient, get_pending_exams
from src.security import audit, guardrails


class AssistantState(TypedDict, total=False):
    """Estado compartilhado entre os nos do grafo."""

    # entrada
    question: str
    patient_id: Optional[str]
    k: int
    trace_id: str

    # intermediarios
    sanitized_question: str
    patient_context: str
    pending_exams: List[Dict]
    docs: List
    draft_answer: str

    # saida
    answer: str
    sources: List[Dict]
    alerts: List[Dict]
    violations: List[str]
    requires_human_validation: bool
    blocked: bool
    path: List[str]


def _trace(state: AssistantState, node: str) -> AssistantState:
    """Registra a passagem pelo no, tanto no log quanto no estado."""
    audit.log_event(
        audit.EVENT_NODE,
        state["trace_id"],
        {"no": node},
        patient_id=state.get("patient_id"),
    )
    return {"path": state.get("path", []) + [node]}


# ------------------------------------------------------------------ nos
def triagem(state: AssistantState) -> AssistantState:
    """Guardrail de entrada: sanitiza PII e barra pedidos fora de escopo."""
    update = _trace(state, "triagem")
    gate = guardrails.check_input(state["question"])

    audit.log_event(
        audit.EVENT_INPUT,
        state["trace_id"],
        {"pergunta": gate.text, "violacoes": gate.violations},
        patient_id=state.get("patient_id"),
    )

    update.update({
        "sanitized_question": gate.text,
        "violations": list(gate.violations),
        "blocked": not gate.allowed,
    })
    if not gate.allowed:
        update.update({
            "answer": gate.text,
            "sources": [],
            "alerts": [],
            "requires_human_validation": False,
        })
    return update


def carregar_prontuario(state: AssistantState) -> AssistantState:
    """Consulta a base estruturada (SQLite) do paciente, se houver."""
    update = _trace(state, "carregar_prontuario")
    patient_id = state.get("patient_id")

    if not patient_id:
        update["patient_context"] = NO_PATIENT
        return update

    patient = get_patient(patient_id)
    if patient is None:
        audit.log_event(
            audit.EVENT_ERROR,
            state["trace_id"],
            {"motivo": "paciente_inexistente"},
            patient_id=patient_id,
        )
        update.update({
            "blocked": True,
            "answer": f"Paciente {patient_id} nao encontrado na base.",
            "sources": [],
            "alerts": [],
            "patient_context": NO_PATIENT,
        })
        return update

    audit.log_event(
        audit.EVENT_PATIENT,
        state["trace_id"],
        {"diagnostico": patient["diagnostico"],
         "alergias": patient["alergias"]},
        patient_id=patient_id,
    )
    update["patient_context"] = format_patient(patient)
    return update


def verificar_exames(state: AssistantState) -> AssistantState:
    """Levanta os exames pendentes: define o desvio condicional do fluxo."""
    update = _trace(state, "verificar_exames")
    patient_id = state.get("patient_id")
    update["pending_exams"] = (
        get_pending_exams(patient_id) if patient_id else []
    )
    return update


def alerta_exames(state: AssistantState) -> AssistantState:
    """Emite alerta: ha exames que podem mudar a conduta sugerida."""
    update = _trace(state, "alerta_exames")
    pendentes = state.get("pending_exams", [])
    nomes = [e["nome"] for e in pendentes]

    alerta = {
        "nivel": "atencao",
        "tipo": "exames_pendentes",
        "mensagem": (
            "Conduta sugerida com informacao incompleta: "
            f"{len(nomes)} exame(s) pendente(s) - {', '.join(nomes)}."
        ),
    }
    audit.log_event(
        audit.EVENT_ALERT, state["trace_id"], alerta,
        patient_id=state.get("patient_id"),
    )
    update["alerts"] = state.get("alerts", []) + [alerta]
    return update


def buscar_evidencia(state: AssistantState) -> AssistantState:
    """Recupera a evidencia cientifica no banco vetorial (RAG)."""
    from src.rag.vectorstore import get_retriever

    update = _trace(state, "buscar_evidencia")
    k = state.get("k", RETRIEVER_TOP_K)
    docs = get_retriever(k).invoke(state["sanitized_question"])

    audit.log_event(
        audit.EVENT_RETRIEVAL,
        state["trace_id"],
        {"k": k, "pmids": [d.metadata.get("pmid") for d in docs]},
        patient_id=state.get("patient_id"),
    )
    update["docs"] = docs
    return update


def sugerir_conduta(state: AssistantState) -> AssistantState:
    """Gera a sugestao com a LLM customizada, ancorada em evidencia."""
    from langchain_core.output_parsers import StrOutputParser

    from src.app.llm import get_llm

    update = _trace(state, "sugerir_conduta")
    docs = state.get("docs", [])

    chain = ASSISTANT_PROMPT | get_llm() | StrOutputParser()
    draft = chain.invoke({
        "question": state["sanitized_question"],
        "patient_context": state.get("patient_context", NO_PATIENT),
        "evidence": format_evidence(docs),
    })

    audit.log_event(
        audit.EVENT_LLM, state["trace_id"], {"caracteres": len(draft)},
        patient_id=state.get("patient_id"),
    )
    update["draft_answer"] = draft.strip()
    return update


def guardrail(state: AssistantState) -> AssistantState:
    """Valida a sugestao fora do modelo antes de qualquer exibicao."""
    update = _trace(state, "guardrail")

    checked = guardrails.validate_answer(
        state.get("draft_answer", ""),
        has_evidence=bool(state.get("docs")),
        trace_id=state["trace_id"],
        patient_id=state.get("patient_id"),
    )

    update.update({
        "answer": checked.text,
        "sources": build_sources(state.get("docs", []),
                                 state.get("patient_id")),
        "violations": state.get("violations", []) + checked.violations,
        "requires_human_validation": checked.requires_human_validation,
    })
    return update


def alertar_equipe(state: AssistantState) -> AssistantState:
    """Consolida os alertas destinados a equipe medica."""
    update = _trace(state, "alertar_equipe")
    alerts = list(state.get("alerts", []))
    answer = state.get("answer", "")
    patient_id = state.get("patient_id")

    # Alerta critico: a conduta menciona substancia alergenica do paciente.
    if patient_id:
        patient = get_patient(patient_id)
        for substancia in (patient or {}).get("alergias", []):
            if substancia.lower() in answer.lower():
                alerts.append({
                    "nivel": "critico",
                    "tipo": "alergia",
                    "mensagem": (
                        f"A resposta menciona '{substancia}', substancia a "
                        f"qual o paciente {patient_id} tem alergia "
                        "registrada."
                    ),
                })

    if "dose_na_saida" in state.get("violations", []) or \
            "frequencia_na_saida" in state.get("violations", []):
        alerts.append({
            "nivel": "critico",
            "tipo": "tentativa_de_prescricao",
            "mensagem": ("O modelo produziu posologia; o conteudo foi "
                         "redigido automaticamente e exige prescricao "
                         "de profissional habilitado."),
        })

    if "sem_citacao" in state.get("violations", []):
        alerts.append({
            "nivel": "atencao",
            "tipo": "sem_fonte",
            "mensagem": ("Resposta sem citacao de PMID: nao foi possivel "
                         "rastrear a fonte da afirmacao."),
        })

    alerts.append({
        "nivel": "informativo",
        "tipo": "validacao_humana",
        "mensagem": ("Sugestao pendente de validacao por profissional de "
                     "saude responsavel."),
    })

    for a in alerts[len(state.get("alerts", [])):]:
        audit.log_event(
            audit.EVENT_ALERT, state["trace_id"], a, patient_id=patient_id,
        )

    update["alerts"] = alerts
    return update


# --------------------------------------------------------- roteamento
def rota_triagem(state: AssistantState) -> str:
    return "bloqueado" if state.get("blocked") else "ok"


def rota_prontuario(state: AssistantState) -> str:
    return "bloqueado" if state.get("blocked") else "ok"


def rota_exames(state: AssistantState) -> str:
    return "pendentes" if state.get("pending_exams") else "nenhum"


# ------------------------------------------------------------- grafo
def build_graph():
    """Monta e compila o grafo do fluxo clinico."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(AssistantState)

    g.add_node("triagem", triagem)
    g.add_node("carregar_prontuario", carregar_prontuario)
    g.add_node("verificar_exames", verificar_exames)
    g.add_node("alerta_exames", alerta_exames)
    g.add_node("buscar_evidencia", buscar_evidencia)
    g.add_node("sugerir_conduta", sugerir_conduta)
    g.add_node("guardrail", guardrail)
    g.add_node("alertar_equipe", alertar_equipe)

    g.add_edge(START, "triagem")
    g.add_conditional_edges(
        "triagem", rota_triagem,
        {"ok": "carregar_prontuario", "bloqueado": END},
    )
    g.add_conditional_edges(
        "carregar_prontuario", rota_prontuario,
        {"ok": "verificar_exames", "bloqueado": END},
    )
    g.add_conditional_edges(
        "verificar_exames", rota_exames,
        {"pendentes": "alerta_exames", "nenhum": "buscar_evidencia"},
    )
    g.add_edge("alerta_exames", "buscar_evidencia")
    g.add_edge("buscar_evidencia", "sugerir_conduta")
    g.add_edge("sugerir_conduta", "guardrail")
    g.add_edge("guardrail", "alertar_equipe")
    g.add_edge("alertar_equipe", END)

    return g.compile()


_GRAPH = None


def get_graph():
    """Compila o grafo uma unica vez por processo."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_flow(
    question: str,
    patient_id: Optional[str] = None,
    k: int = RETRIEVER_TOP_K,
    trace_id: Optional[str] = None,
) -> AssistantState:
    """Executa o atendimento completo e devolve o estado final."""
    state: AssistantState = {
        "question": question,
        "patient_id": patient_id,
        "k": k,
        "trace_id": trace_id or audit.new_trace_id(),
        "alerts": [],
        "violations": [],
        "path": [],
        "blocked": False,
    }
    return get_graph().invoke(state)


def export_diagram(path=None) -> str:
    """Exporta o diagrama do fluxo em Mermaid (usado no relatorio)."""
    from src.config import DOCS_DIR

    path = path or DOCS_DIR / "fluxo_langgraph.mmd"
    mermaid = get_graph().get_graph().draw_mermaid()
    path.write_text(mermaid, encoding="utf-8")
    print(f"[ok] diagrama -> {path}")
    return mermaid


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description="Fluxo clinico (LangGraph)")
    p.add_argument("--question", default="Qual a conduta indicada para este paciente?")
    p.add_argument("--patient", default=None)
    p.add_argument("--diagram", action="store_true",
                   help="apenas exporta o diagrama, sem executar a LLM")
    args = p.parse_args()

    if args.diagram:
        print(export_diagram())
        raise SystemExit(0)

    final = run_flow(args.question, args.patient)

    print("\ncaminho:", " -> ".join(final["path"]))
    print("\n--- RESPOSTA ---\n" + final.get("answer", ""))
    print("\n--- ALERTAS ---")
    for a in final.get("alerts", []):
        print(f"  [{a['nivel']}] {a['mensagem']}")
    print("\n--- FONTES ---")
    print(json.dumps(final.get("sources", []), indent=2, ensure_ascii=False))
