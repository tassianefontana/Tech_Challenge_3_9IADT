"""Etapa 3 - Avaliacao do modelo: base vs fine-tuned.

Metricas:
  - Acuracia e F1 macro do veredito (yes/no/maybe). F1 macro e a metrica
    principal porque o dataset e desbalanceado (apenas ~11% de `maybe`).
  - ROUGE-L da justificativa em relacao a conclusao de referencia.
  - Taxa de respostas mal formatadas (sem veredito parseavel).

Uso:
    python -m src.finetuning.evaluate                  # base + fine-tuned
    python -m src.finetuning.evaluate --only base
    python -m src.finetuning.evaluate --limit 30       # smoke test rapido
"""

import argparse
import json
import re
from typing import Dict, List, Optional

from src.config import (
    ADAPTER_DIR,
    BASE_LLM,
    EVAL_RESULTS_FILE,
    GENERATION_ARGS,
    TEST_FILE,
    VALID_LABELS,
    VERDICT_PREFIX,
)

_VERDICT_RE = re.compile(
    rf"{re.escape(VERDICT_PREFIX)}\s*(yes|no|maybe)", re.IGNORECASE
)
_FALLBACK_RE = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)


# ---------------------------------------------------------------- parsing
def parse_verdict(text: str) -> Optional[str]:
    """Extrai o veredito. Tenta o formato estruturado; senao, o 1o token valido."""
    m = _VERDICT_RE.search(text)
    if m:
        return m.group(1).lower()
    m = _FALLBACK_RE.search(text)
    return m.group(1).lower() if m else None


def parse_rationale(text: str) -> str:
    parts = re.split(r"Justificativa:\s*", text, maxsplit=1)
    return (parts[1] if len(parts) > 1 else text).strip()


# ---------------------------------------------------------------- metricas
def macro_f1(y_true: List[str], y_pred: List[Optional[str]]) -> Dict[str, float]:
    """F1 por classe e F1 macro, sem depender de sklearn."""
    per_class: Dict[str, float] = {}
    for label in sorted(VALID_LABELS):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = (
            2 * prec * rec / (prec + rec) if prec + rec else 0.0
        )
    per_class["macro"] = sum(
        per_class[l] for l in VALID_LABELS
    ) / len(VALID_LABELS)
    return {k: round(v, 4) for k, v in per_class.items()}


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure via LCS sobre tokens (implementacao propria)."""
    pred = prediction.lower().split()
    ref = reference.lower().split()
    if not pred or not ref:
        return 0.0

    # LCS com duas linhas (memoria O(min(n,m)))
    prev = [0] * (len(ref) + 1)
    for a in pred:
        cur = [0] * (len(ref) + 1)
        for j, b in enumerate(ref, 1):
            cur[j] = prev[j - 1] + 1 if a == b else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]

    if lcs == 0:
        return 0.0
    prec = lcs / len(pred)
    rec = lcs / len(ref)
    return 2 * prec * rec / (prec + rec)


def confusion(y_true: List[str], y_pred: List[Optional[str]]) -> Dict[str, Dict]:
    labels = sorted(VALID_LABELS)
    matrix = {t: {p: 0 for p in labels + ["unparsed"]} for t in labels}
    for t, p in zip(y_true, y_pred):
        matrix[t][p if p in VALID_LABELS else "unparsed"] += 1
    return matrix


# ---------------------------------------------------------------- inferencia
def load_test_set(limit: Optional[int]) -> List[dict]:
    rows = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def generate_all(model_id: str, adapter: Optional[str], samples: List[dict],
                 batch_size: int) -> List[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # obrigatorio para geracao em batch

    kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()

    model.eval()

    gen_kwargs = {k: v for k, v in GENERATION_ARGS.items() if v is not None}
    outputs: List[str] = []

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                s["messages"][:-1],  # system + user, sem a resposta
                tokenize=False,
                add_generation_prompt=True,
            )
            for s in batch
        ]
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            ids = model.generate(
                **enc, pad_token_id=tokenizer.pad_token_id, **gen_kwargs
            )

        for j in range(len(batch)):
            new_tokens = ids[j][enc["input_ids"].shape[1] :]
            outputs.append(
                tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            )

        print(f"  [{min(i + batch_size, len(samples))}/{len(samples)}]", end="\r")

    print()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def score(samples: List[dict], predictions: List[str]) -> Dict:
    y_true = [s["label"] for s in samples]
    y_pred = [parse_verdict(p) for p in predictions]

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    unparsed = sum(1 for p in y_pred if p is None)

    rouge = [
        rouge_l(
            parse_rationale(pred),
            parse_rationale(s["messages"][-1]["content"]),
        )
        for s, pred in zip(samples, predictions)
    ]

    return {
        "n": len(samples),
        "accuracy": round(correct / len(samples), 4),
        "f1": macro_f1(y_true, y_pred),
        "rouge_l": round(sum(rouge) / len(rouge), 4),
        "taxa_nao_parseada": round(unparsed / len(samples), 4),
        "matriz_confusao": confusion(y_true, y_pred),
    }


# ---------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Avaliacao base vs fine-tuned")
    p.add_argument("--model", default=BASE_LLM)
    p.add_argument("--adapter", default=str(ADAPTER_DIR))
    p.add_argument("--only", choices=["base", "finetuned"], default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


def run() -> None:
    from pathlib import Path

    args = parse_args()
    samples = load_test_set(args.limit)
    print(f"[info] avaliando {len(samples)} exemplos do test set\n")

    results: Dict[str, Dict] = {}

    if args.only != "finetuned":
        print("[1/2] modelo BASE (sem fine-tuning)...")
        preds = generate_all(args.model, None, samples, args.batch_size)
        results["base"] = score(samples, preds)
        results["base"]["exemplos"] = preds[:3]

    if args.only != "base":
        if not Path(args.adapter).exists():
            print(f"[aviso] adaptador {args.adapter} nao encontrado; pulando.")
        else:
            print("[2/2] modelo FINE-TUNED (QLoRA)...")
            preds = generate_all(
                args.model, args.adapter, samples, args.batch_size
            )
            results["finetuned"] = score(samples, preds)
            results["finetuned"]["exemplos"] = preds[:3]

    EVAL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 62)
    print(f"{'metrica':<22}{'base':>18}{'fine-tuned':>18}")
    print("=" * 62)
    rows = [
        ("acuracia", lambda r: r["accuracy"]),
        ("f1 macro", lambda r: r["f1"]["macro"]),
        ("f1 (yes)", lambda r: r["f1"]["yes"]),
        ("f1 (no)", lambda r: r["f1"]["no"]),
        ("f1 (maybe)", lambda r: r["f1"]["maybe"]),
        ("rouge-l", lambda r: r["rouge_l"]),
        ("nao parseada", lambda r: r["taxa_nao_parseada"]),
    ]
    for name, fn in rows:
        b = f"{fn(results['base']):.4f}" if "base" in results else "-"
        t = f"{fn(results['finetuned']):.4f}" if "finetuned" in results else "-"
        print(f"{name:<22}{b:>18}{t:>18}")
    print("=" * 62)
    print(f"\n[ok] resultados salvos em {EVAL_RESULTS_FILE}")


if __name__ == "__main__":
    run()
