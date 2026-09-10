"""Etapa 3 - Fine-tuning QLoRA do assistente medico.

Carrega o modelo base em 4-bit (NF4) e treina apenas adaptadores LoRA,
o que permite rodar em uma GPU T4 de 16 GB (Google Colab gratuito).

Uso (Colab ou maquina com GPU NVIDIA):
    python -m src.finetuning.train
    python -m src.finetuning.train --epochs 2 --model Qwen/Qwen2.5-0.5B-Instruct

Em CPU o treino nao e viavel; o script aborta com mensagem explicita.
"""

import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from src.config import (
    ADAPTER_DIR,
    BASE_LLM,
    LORA_CONFIG,
    MAX_SEQ_LENGTH,
    TRAIN_FILE,
    TRAINING_ARGS,
    VAL_FILE,
)


def load_jsonl_dataset(path) -> Dataset:
    """Carrega o JSONL gerado na Etapa 2 mantendo apenas o campo `messages`.

    O SFTTrainer detecta o formato conversacional e aplica o chat template
    do tokenizer automaticamente.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append({"messages": json.loads(line)["messages"]})
    return Dataset.from_list(rows)


def build_model(model_name: str):
    """Carrega o modelo base quantizado em 4-bit e acopla os adaptadores LoRA."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # incompativel com gradient checkpointing
    model.config.pretraining_tp = 1

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
    model.print_trainable_parameters()
    return model


def build_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # correto para treino (left apenas p/ geracao)
    return tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tuning QLoRA - PubMedQA")
    p.add_argument("--model", default=BASE_LLM, help="modelo base no HuggingFace")
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--output", default=str(ADAPTER_DIR))
    p.add_argument(
        "--max-seq-length", type=int, default=MAX_SEQ_LENGTH,
        help="reduza para 1024 se ocorrer OOM na T4",
    )
    return p.parse_args()


def run() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "GPU NVIDIA nao detectada. O fine-tuning em 4-bit exige CUDA.\n"
            "Use o Google Colab: notebooks/finetuning_colab.ipynb "
            "(Ambiente de execucao > Alterar tipo > GPU T4)."
        )

    print(f"[info] GPU: {torch.cuda.get_device_name(0)}")

    train_ds = load_jsonl_dataset(TRAIN_FILE)
    val_ds = load_jsonl_dataset(VAL_FILE)
    print(f"[info] train={len(train_ds)} | val={len(val_ds)}")

    tokenizer = build_tokenizer(args.model)
    model = build_model(args.model)

    cfg = dict(TRAINING_ARGS)
    if args.epochs is not None:
        cfg["num_train_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["per_device_train_batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["learning_rate"] = args.lr

    sft_config = SFTConfig(
        output_dir=str(args.output),
        max_length=args.max_seq_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        # Treina apenas nos tokens da resposta do assistente, nao no prompt.
        # Melhora a qualidade do veredito e evita o modelo aprender a copiar
        # o contexto de entrada.
        assistant_only_loss=True,
        **cfg,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))

    metrics = trainer.evaluate()
    with open(f"{args.output}/train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[ok] adaptador LoRA salvo em {args.output}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run()
