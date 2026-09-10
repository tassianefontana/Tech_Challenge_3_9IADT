"""Etapa 2 - Pipeline de preprocessing, anonimizacao e curadoria.

Le o PubMedQA bruto (ori_pqal.json) e produz um dataset curado em
data/processed/pubmedqa_processed.json, junto com um relatorio de
estatisticas usado no relatorio tecnico.

Uso:
    python -m src.preprocessing.build_dataset
"""

import hashlib
import json
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional

from src.config import (
    MAX_CONTEXT_CHARS,
    MIN_ANSWER_CHARS,
    MIN_CONTEXT_CHARS,
    PROCESSED_FILE,
    PUBMEDQA_RAW_FILE,
    STATS_FILE,
    VALID_LABELS,
)
from src.preprocessing.anonymizer import anonymize

_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(text: str) -> str:
    """Normaliza unicode, remove caracteres de controle e espacos extras."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _CTRL.sub(" ", text)
    text = _WS.sub(" ", text)
    return text.strip()


def build_context(item: dict) -> str:
    """Concatena os contextos preservando os rotulos de secao do abstract.

    O PubMedQA traz CONTEXTS (lista de paragrafos) e LABELS (BACKGROUND,
    METHODS, RESULTS...). Manter os rotulos ajuda o modelo a distinguir
    metodologia de conclusao.
    """
    contexts = item.get("CONTEXTS") or []
    labels = (item.get("LABELS") or [])[: len(contexts)]

    parts: List[str] = []
    for i, ctx in enumerate(contexts):
        ctx = clean_text(ctx)
        if not ctx:
            continue
        if i < len(labels) and labels[i]:
            parts.append(f"{clean_text(labels[i]).upper()}: {ctx}")
        else:
            parts.append(ctx)
    return "\n".join(parts)


def curate(record: dict, stats: Counter) -> Optional[dict]:
    """Aplica as regras de curadoria. Retorna None se o exemplo for descartado."""
    if not record["question"]:
        stats["descartado_sem_pergunta"] += 1
        return None

    if record["label"] not in VALID_LABELS:
        stats["descartado_label_invalido"] += 1
        return None

    if len(record["answer"]) < MIN_ANSWER_CHARS:
        stats["descartado_resposta_curta"] += 1
        return None

    if len(record["context"]) < MIN_CONTEXT_CHARS:
        stats["descartado_contexto_curto"] += 1
        return None

    if len(record["context"]) > MAX_CONTEXT_CHARS:
        record["context"] = record["context"][:MAX_CONTEXT_CHARS].rsplit(" ", 1)[0]
        stats["contexto_truncado"] += 1

    return record


def fingerprint(record: dict) -> str:
    """Hash usado para deduplicacao (pergunta + resposta normalizadas)."""
    key = f"{record['question'].lower()}||{record['answer'].lower()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def run() -> List[dict]:
    if not PUBMEDQA_RAW_FILE.exists():
        raise FileNotFoundError(
            f"{PUBMEDQA_RAW_FILE} nao encontrado. "
            "Rode: python -m src.preprocessing.download_dataset"
        )

    with open(PUBMEDQA_RAW_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    stats: Counter = Counter()
    pii_hits: Counter = Counter()
    label_dist: Counter = Counter()
    seen: Dict[str, str] = {}
    dataset: List[dict] = []

    for pmid, item in raw.items():
        stats["total_bruto"] += 1

        question = clean_text(item.get("QUESTION", ""))
        context = build_context(item)
        answer = clean_text(item.get("LONG_ANSWER", ""))
        label = clean_text(item.get("final_decision", "")).lower()

        # --- Anonimizacao (aplicada aos tres campos textuais) ---
        question, h1 = anonymize(question)
        context, h2 = anonymize(context)
        answer, h3 = anonymize(answer)
        for h in (h1, h2, h3):
            pii_hits.update(h)

        record = {
            "id": pmid,
            "question": question,
            "context": context,
            "answer": answer,
            "label": label,
        }

        record = curate(record, stats)
        if record is None:
            continue

        # --- Deduplicacao ---
        fp = fingerprint(record)
        if fp in seen:
            stats["descartado_duplicado"] += 1
            continue
        seen[fp] = pmid

        record["n_chars_context"] = len(record["context"])
        label_dist[record["label"]] += 1
        dataset.append(record)

    stats["total_final"] = len(dataset)

    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    report = {
        "arquivo_origem": PUBMEDQA_RAW_FILE.name,
        "curadoria": dict(stats),
        "distribuicao_labels": dict(label_dist),
        "pii_substituida": dict(pii_hits),
        "media_chars_contexto": (
            round(sum(r["n_chars_context"] for r in dataset) / len(dataset), 1)
            if dataset
            else 0
        ),
    }
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[ok] {len(dataset)} exemplos curados -> {PROCESSED_FILE}")
    print(f"[ok] estatisticas -> {STATS_FILE}")
    return dataset


if __name__ == "__main__":
    run()
