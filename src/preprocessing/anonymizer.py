"""Anonimizacao / de-identificacao de texto clinico.

Requisito explicito do Tech Challenge Fase 3: "Preparar os dados com
tecnicas de preprocessing, anonimizacao e curadoria."

Estrategia: substituicao por placeholders tipados (pseudonimizacao),
preservando a semantica clinica do texto para o modelo.
"""

import re
from typing import Dict, Tuple

# Ordem importa: padroes mais especificos primeiro.
PATTERNS: Dict[str, re.Pattern] = {
    # Identificadores diretos
    "[EMAIL]": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "[URL]": re.compile(r"\bhttps?://\S+|\bwww\.\S+"),
    "[CPF]": re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    "[CNS]": re.compile(r"\b\d{3}\s?\d{4}\s?\d{4}\s?\d{4}\b"),
    "[PHONE]": re.compile(
        r"(?<!\d)(?:\+\d{1,3}[\s-]?)?\(?\d{2,3}\)?[\s.-]?\d{4,5}[\s.-]?\d{4}(?!\d)"
    ),
    # Identificadores de prontuario / registro hospitalar
    "[MRN]": re.compile(
        r"\b(?:MRN|prontu[aá]rio|registro|matr[ií]cula|paciente)\s*[:#-]?\s*\d{4,}\b",
        re.IGNORECASE,
    ),
    # Datas completas (dd/mm/aaaa, aaaa-mm-dd, 12 Jan 2020)
    "[DATE]": re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
        r"|\d{4}[/-]\d{1,2}[/-]\d{1,2}"
        r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b",
        re.IGNORECASE,
    ),
    # Nomes proprios precedidos de titulo clinico
    "[NAME]": re.compile(
        r"\b(?:Dr|Dra|Doutor|Doutora|Prof|Mr|Mrs|Ms|Sr|Sra)\.?\s+"
        r"(?:[A-Z][a-zA-ZÀ-ÿ'-]+\s?){1,3}"
    ),
}

# Idades >= 90 sao identificadores indiretos sob HIPAA Safe Harbor.
_AGE_90_PLUS = re.compile(
    r"\b(9\d|1\d{2})[\s-]*(?:years?[\s-]*old|anos)\b", re.IGNORECASE
)


def anonymize(text: str) -> Tuple[str, Dict[str, int]]:
    """Aplica a de-identificacao e retorna (texto, contagem por categoria)."""
    if not text:
        return "", {}

    hits: Dict[str, int] = {}

    for placeholder, pattern in PATTERNS.items():
        text, n = pattern.subn(placeholder, text)
        if n:
            hits[placeholder] = hits.get(placeholder, 0) + n

    text, n = _AGE_90_PLUS.subn("[AGE_90+]", text)
    if n:
        hits["[AGE_90+]"] = n

    return text, hits


def contains_pii(text: str) -> bool:
    if any(p.search(text) for p in PATTERNS.values()):
        return True
    return bool(_AGE_90_PLUS.search(text))
