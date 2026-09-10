"""Etapa 2 - Split estratificado train/val/test em formato de chat (JSONL).

O formato de saida e o consumido diretamente pelo SFTTrainer (trl):
    {"id": ..., "label": ..., "messages": [system, user, assistant]}
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from src.config import (
    INSTRUCTION_TEMPLATE,
    PROCESSED_FILE,
    RANDOM_SEED,
    RATIONALE_PREFIX,
    SPLIT_RATIOS,
    SYSTEM_PROMPT,
    TEST_FILE,
    TRAIN_FILE,
    VAL_FILE,
    VERDICT_PREFIX,
)


def to_chat(record: dict) -> dict:
    """Converte um exemplo curado no formato conversacional de treino."""
    user = INSTRUCTION_TEMPLATE.format(
        question=record["question"], context=record["context"]
    )
    assistant = (
        f"{VERDICT_PREFIX} {record['label']}\n"
        f"{RATIONALE_PREFIX} {record['answer']}"
    )

    return {
        "id": record["id"],
        "label": record["label"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def stratified_split(data: List[dict]) -> Dict[str, List[dict]]:
    """Split estratificado por label, preservando a distribuicao yes/no/maybe."""
    rng = random.Random(RANDOM_SEED)

    by_label = defaultdict(list)
    for rec in data:
        by_label[rec["label"]].append(rec)

    splits: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}

    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        splits["train"] += items[:n_train]
        splits["val"] += items[n_train : n_train + n_val]
        splits["test"] += items[n_train + n_val :]

    for name in splits:
        rng.shuffle(splits[name])

    return splits


def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(to_chat(rec), ensure_ascii=False) + "\n")


def run() -> None:
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"{PROCESSED_FILE} nao encontrado. "
            "Rode: python -m src.preprocessing.build_dataset"
        )

    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    splits = stratified_split(data)
    targets = {"train": TRAIN_FILE, "val": VAL_FILE, "test": TEST_FILE}

    for name, path in targets.items():
        write_jsonl(path, splits[name])
        dist = defaultdict(int)
        for r in splits[name]:
            dist[r["label"]] += 1
        dist_str = ", ".join(f"{k}={v}" for k, v in sorted(dist.items()))
        print(f"[ok] {name:5s} {len(splits[name]):5d} exemplos ({dist_str}) -> {path}")


if __name__ == "__main__":
    run()
