"""Interface de demonstracao (Streamlit) do assistente medico.

Executar:
    streamlit run src/app/ui.py

Mostra, numa unica tela, os quatro pontos exigidos no video de entrega:
funcionamento da LLM personalizada, execucao do fluxo automatizado,
resposta clinica contextualizada e os logs/validacao das respostas.
"""

import json
import sys
from pathlib import Path

# Permite `streamlit run src/app/ui.py` sem instalar o pacote.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st  # noqa: E402

from src.app.llm import adapter_available  # noqa: E402
from src.config import ADAPTER_DIR, BASE_LLM, RETRIEVER_TOP_K  # noqa: E402
from src.db.patients import list_patients  # noqa: E402
from src.security import audit  # noqa: E402
from src.workflow.graph import run_flow  # noqa: E402

NIVEL_UI = {
    "critico": ("error", "CRITICO"),
    "atencao": ("warning", "ATENCAO"),
    "informativo": ("info", "INFO"),
}

EXEMPLOS = [
    "Quais condutas a evidencia sustenta para este quadro?",
    "Ha evidencia de beneficio do tratamento antibiotico precoce?",
    "O que a literatura indica sobre o prognostico deste diagnostico?",
]

st.set_page_config(page_title="Assistente Medico - Fase 3", layout="wide")


@st.cache_data(show_spinner=False)
def carregar_pacientes():
    return list_patients()


# ------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Configuracao")

    if adapter_available():
        st.success("LLM fine-tunada carregada")
        st.caption(f"{BASE_LLM}\n+ LoRA: {ADAPTER_DIR.name}")
    else:
        st.warning("Adaptador LoRA ausente: usando o modelo base")

    pacientes = carregar_pacientes()
    opcoes = ["(sem paciente)"] + [
        f"{p['id']} - {p['iniciais']}, {p['idade']}a, {p['diagnostico']}"
        for p in pacientes
    ]
    escolha = st.selectbox("Paciente", opcoes)
    patient_id = None if escolha.startswith("(") else escolha.split(" - ")[0]

    k = st.slider("Trechos de evidencia (k)", 1, 8, RETRIEVER_TOP_K)

    st.divider()
    st.caption(
        "Sistema de apoio a decisao. Nao prescreve: toda sugestao passa "
        "por guardrails programaticos e exige validacao humana."
    )

# -------------------------------------------------------------- corpo
st.title("Assistente Medico - Tech Challenge Fase 3")

pergunta = st.text_area(
    "Pergunta clinica",
    value=EXEMPLOS[0],
    height=90,
    help="A pergunta e sanitizada (PII) antes de chegar ao modelo.",
)

if st.button("Executar fluxo", type="primary"):
    with st.spinner("Executando o fluxo clinico..."):
        estado = run_flow(pergunta, patient_id=patient_id, k=k)
    st.session_state["estado"] = estado

estado = st.session_state.get("estado")

if estado:
    st.caption(f"trace_id: `{estado['trace_id']}`")

    # ---------------------------------------------------- fluxo
    st.subheader("Fluxo executado")
    st.code(" -> ".join(estado.get("path", [])), language="text")

    # ---------------------------------------------------- alertas
    alertas = estado.get("alerts", [])
    if alertas:
        st.subheader("Alertas para a equipe medica")
        for a in alertas:
            fn_name, rotulo = NIVEL_UI.get(a["nivel"], ("info", "INFO"))
            getattr(st, fn_name)(f"**{rotulo} - {a['tipo']}**  \n{a['mensagem']}")

    # ---------------------------------------------------- resposta
    st.subheader("Resposta")
    st.markdown(estado.get("answer", ""))

    violacoes = estado.get("violations", [])
    if violacoes:
        st.warning("Guardrails acionados: " + ", ".join(sorted(set(violacoes))))
    if estado.get("requires_human_validation"):
        st.info("Conteudo pendente de validacao por profissional de saude.")

    # ---------------------------------------------------- fontes
    fontes = estado.get("sources", [])
    if fontes:
        st.subheader("Fontes utilizadas (explainability)")
        for f in fontes:
            if f["tipo"] == "evidencia_cientifica":
                with st.expander(
                    f"PMID {f['pmid']} - conclusao: {f['conclusao']}"
                ):
                    st.write(f["trecho"] + "...")
                    st.markdown(f"[Abrir no PubMed]({f['url']})")
            else:
                st.caption(
                    f"Prontuario do paciente {f['paciente_id']} "
                    f"(fonte: {f['origem']})"
                )

    # ---------------------------------------------------- auditoria
    st.subheader("Trilha de auditoria")
    eventos = audit.read_events(trace_id=estado["trace_id"])
    st.caption(f"{len(eventos)} evento(s) registrados nesta execucao")
    st.json([
        {"timestamp": e["timestamp"], "event": e["event"], **e["detail"]}
        for e in eventos
    ])

    with st.expander("Linhas brutas do log (logs/audit.log)"):
        st.code(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in eventos),
            language="json",
        )
else:
    st.info("Selecione um paciente, escreva a pergunta e execute o fluxo.")
