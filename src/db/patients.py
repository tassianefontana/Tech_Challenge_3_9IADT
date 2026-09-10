"""Base estruturada de pacientes (SQLite) com dados sinteticos.

O PubMedQA fornece apenas evidencia cientifica, nao prontuarios. Esta base
simula os registros internos do hospital para que o assistente possa
contextualizar as respostas com informacoes atualizadas do paciente,
conforme exigido no desafio.

Todos os dados sao gerados sinteticamente com seed fixa: nenhum dado real
de paciente e utilizado.
"""

import json
import random
import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from typing import Dict, List, Optional

from src.config import PATIENTS_DB, PATIENTS_SEED_FILE, RANDOM_SEED

SCHEMA = """
CREATE TABLE IF NOT EXISTS pacientes (
    id              TEXT PRIMARY KEY,
    iniciais        TEXT NOT NULL,
    idade           INTEGER NOT NULL,
    sexo            TEXT NOT NULL,
    setor           TEXT NOT NULL,
    data_internacao TEXT NOT NULL,
    diagnostico     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS exames (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id TEXT NOT NULL REFERENCES pacientes(id),
    nome        TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('pendente','concluido')),
    resultado   TEXT,
    solicitado_em TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS medicacoes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id TEXT NOT NULL REFERENCES pacientes(id),
    nome        TEXT NOT NULL,
    posologia   TEXT NOT NULL,
    ativo       INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS alergias (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id TEXT NOT NULL REFERENCES pacientes(id),
    substancia  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_exames_paciente ON exames(paciente_id);
CREATE INDEX IF NOT EXISTS idx_medicacoes_paciente ON medicacoes(paciente_id);
"""

DIAGNOSTICOS = [
    ("Insuficiencia cardiaca descompensada", ["Ecocardiograma", "BNP", "Troponina"]),
    ("Pneumonia adquirida na comunidade", ["Radiografia de torax", "Hemocultura", "PCR"]),
    ("Sepse de foco urinario", ["Urocultura", "Lactato serico", "Hemograma"]),
    ("Doenca pulmonar obstrutiva cronica exacerbada", ["Gasometria arterial", "Espirometria"]),
    ("Acidente vascular cerebral isquemico", ["Tomografia de cranio", "Angio-TC", "Glicemia"]),
    ("Diabetes mellitus tipo 2 descompensado", ["Hemoglobina glicada", "Cetonemia", "Funcao renal"]),
    ("Ventilacao mecanica prolongada", ["Gasometria arterial", "Radiografia de torax"]),
]

MEDICACOES = [
    ("Enoxaparina", "40 mg SC 1x/dia"),
    ("Omeprazol", "40 mg IV 1x/dia"),
    ("Ceftriaxona", "1 g IV 12/12h"),
    ("Furosemida", "20 mg IV 8/8h"),
    ("Insulina regular", "conforme glicemia capilar"),
    ("Salbutamol", "inalatorio 6/6h"),
]

ALERGIAS = ["Penicilina", "Dipirona", "Iodo", "Sulfa", "Latex"]
SETORES = ["UTI", "Enfermaria", "Semi-intensiva", "Pronto-socorro"]
RESULTADOS = ["dentro da normalidade", "levemente alterado", "alterado"]


# --------------------------------------------------------------- conexao
@contextmanager
def connect(db_path=None):
    """Conexao SQLite com row_factory de dicionario e commit automatico."""
    conn = sqlite3.connect(str(db_path or PATIENTS_DB))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------- seed
def generate_seed(n_patients: int = 20) -> List[dict]:
    """Gera os registros sinteticos de forma deterministica."""
    rng = random.Random(RANDOM_SEED)
    hoje = date(2025, 6, 1)
    letras = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    pacientes = []
    for i in range(1, n_patients + 1):
        diagnostico, exames_dx = rng.choice(DIAGNOSTICOS)
        internacao = hoje - timedelta(days=rng.randint(0, 14))

        exames = []
        for nome in exames_dx:
            pendente = rng.random() < 0.35
            exames.append({
                "nome": nome,
                "status": "pendente" if pendente else "concluido",
                "resultado": None if pendente else rng.choice(RESULTADOS),
                "solicitado_em": (internacao + timedelta(days=rng.randint(0, 2))).isoformat(),
            })

        meds = rng.sample(MEDICACOES, rng.randint(1, 3))

        pacientes.append({
            "id": f"PAC-{i:04d}",
            "iniciais": rng.choice(letras) + "." + rng.choice(letras) + ".",
            "idade": rng.randint(28, 89),
            "sexo": rng.choice(["M", "F"]),
            "setor": rng.choice(SETORES),
            "data_internacao": internacao.isoformat(),
            "diagnostico": diagnostico,
            "exames": exames,
            "medicacoes": [{"nome": n, "posologia": p, "ativo": 1} for n, p in meds],
            "alergias": rng.sample(ALERGIAS, rng.randint(0, 2)),
        })

    return pacientes


def init_db(force: bool = False, db_path=None) -> None:
    """Cria o schema e popula a base. Idempotente, salvo `force=True`."""
    path = db_path or PATIENTS_DB

    if force and path.exists():
        path.unlink()

    existed = path.exists()

    with connect(path) as conn:
        conn.executescript(SCHEMA)

        if existed and conn.execute("SELECT COUNT(*) FROM pacientes").fetchone()[0]:
            print(f"[skip] base ja populada em {path}")
            return

        pacientes = generate_seed()

        for p in pacientes:
            conn.execute(
                "INSERT INTO pacientes VALUES (?,?,?,?,?,?,?)",
                (p["id"], p["iniciais"], p["idade"], p["sexo"], p["setor"],
                 p["data_internacao"], p["diagnostico"]),
            )
            conn.executemany(
                "INSERT INTO exames (paciente_id,nome,status,resultado,solicitado_em)"
                " VALUES (?,?,?,?,?)",
                [(p["id"], e["nome"], e["status"], e["resultado"],
                  e["solicitado_em"]) for e in p["exames"]],
            )
            conn.executemany(
                "INSERT INTO medicacoes (paciente_id,nome,posologia,ativo)"
                " VALUES (?,?,?,?)",
                [(p["id"], m["nome"], m["posologia"], m["ativo"])
                 for m in p["medicacoes"]],
            )
            conn.executemany(
                "INSERT INTO alergias (paciente_id,substancia) VALUES (?,?)",
                [(p["id"], a) for a in p["alergias"]],
            )

    # Versiona o seed em JSON para o repositorio (o .db fica fora do Git)
    with open(PATIENTS_SEED_FILE, "w", encoding="utf-8") as f:
        json.dump(pacientes, f, indent=2, ensure_ascii=False)

    print(f"[ok] {len(pacientes)} pacientes sinteticos -> {path}")
    print(f"[ok] seed versionavel -> {PATIENTS_SEED_FILE}")


# --------------------------------------------------------------- consultas
def get_patient(patient_id: str) -> Optional[Dict]:
    """Retorna o prontuario completo do paciente, ou None se nao existir."""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM pacientes WHERE id = ?", (patient_id,)
        ).fetchone()
        if row is None:
            return None

        patient = dict(row)
        patient["exames"] = [
            dict(r) for r in conn.execute(
                "SELECT nome, status, resultado, solicitado_em FROM exames"
                " WHERE paciente_id = ? ORDER BY status DESC, nome",
                (patient_id,),
            )
        ]
        patient["medicacoes"] = [
            dict(r) for r in conn.execute(
                "SELECT nome, posologia FROM medicacoes"
                " WHERE paciente_id = ? AND ativo = 1",
                (patient_id,),
            )
        ]
        patient["alergias"] = [
            r["substancia"] for r in conn.execute(
                "SELECT substancia FROM alergias WHERE paciente_id = ?",
                (patient_id,),
            )
        ]
        return patient


def get_pending_exams(patient_id: str) -> List[Dict]:
    """Exames pendentes: usado pelo no de verificacao do fluxo LangGraph."""
    with connect() as conn:
        return [
            dict(r) for r in conn.execute(
                "SELECT nome, solicitado_em FROM exames"
                " WHERE paciente_id = ? AND status = 'pendente' ORDER BY nome",
                (patient_id,),
            )
        ]


def list_patients() -> List[Dict]:
    with connect() as conn:
        return [
            dict(r) for r in conn.execute(
                "SELECT id, iniciais, idade, sexo, setor, diagnostico"
                " FROM pacientes ORDER BY id"
            )
        ]


def format_patient(patient: Dict) -> str:
    """Serializa o prontuario para injecao no prompt da LLM."""
    exames_ok = [e for e in patient["exames"] if e["status"] == "concluido"]
    exames_pend = [e for e in patient["exames"] if e["status"] == "pendente"]

    linhas = [
        f"Paciente {patient['id']} ({patient['iniciais']}), "
        f"{patient['idade']} anos, sexo {patient['sexo']}",
        f"Setor: {patient['setor']} | Internado em: {patient['data_internacao']}",
        f"Diagnostico: {patient['diagnostico']}",
        "Alergias: " + (", ".join(patient["alergias"]) or "nenhuma registrada"),
    ]

    if exames_ok:
        linhas.append("Exames concluidos: " + "; ".join(
            f"{e['nome']} ({e['resultado']})" for e in exames_ok))
    if exames_pend:
        linhas.append("Exames PENDENTES: " + ", ".join(
            e["nome"] for e in exames_pend))
    if patient["medicacoes"]:
        linhas.append("Medicacoes ativas: " + "; ".join(
            f"{m['nome']} {m['posologia']}" for m in patient["medicacoes"]))

    return "\n".join(linhas)


if __name__ == "__main__":
    import sys

    init_db(force="--force" in sys.argv)
    print()
    print(format_patient(get_patient("PAC-0001")))
