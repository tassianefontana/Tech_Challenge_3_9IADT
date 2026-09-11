"""Configuracao central do projeto: paths, modelos e hiperparametros."""

from pathlib import Path

# --------------------------------------------------------------------
# Diretorios
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TRAINING_DIR = DATA_DIR / "training"

MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
DOCS_DIR = BASE_DIR / "docs"

# Diretorio do banco vetorial
VECTORDB_DIR = BASE_DIR / "chromadb"
VECTORDB_COLLECTION = "pubmedqa"

for _d in (RAW_DIR, PROCESSED_DIR, TRAINING_DIR, MODELS_DIR, LOGS_DIR, DOCS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Dataset PubMedQA
# --------------------------------------------------------------------
PUBMEDQA_RAW_FILE = RAW_DIR / "ori_pqal.json"
PUBMEDQA_RAW_URL = (
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/"
    "master/data/ori_pqal.json"
)

PROCESSED_FILE = PROCESSED_DIR / "pubmedqa_processed.json"
STATS_FILE = PROCESSED_DIR / "preprocessing_stats.json"

TRAIN_FILE = TRAINING_DIR / "train.jsonl"
VAL_FILE = TRAINING_DIR / "val.jsonl"
TEST_FILE = TRAINING_DIR / "test.jsonl"

# --------------------------------------------------------------------
# Curadoria / split
# --------------------------------------------------------------------
RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

MIN_ANSWER_CHARS = 20
MIN_CONTEXT_CHARS = 100
MAX_CONTEXT_CHARS = 6000
VALID_LABELS = {"yes", "no", "maybe"}

# --------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_LLM = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = MODELS_DIR / "qwen2.5-1.5b-pubmedqa-lora"

RETRIEVER_TOP_K = 3

# --------------------------------------------------------------------
# Fine-tuning (QLoRA)
# --------------------------------------------------------------------
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}

TRAINING_ARGS = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    # trl 1.x removeu warmup_ratio. ~150 steps totais (799/16 x 3 epocas),
    # portanto 5 steps equivalem aos 3% de warmup pretendidos.
    "warmup_steps": 5,
    "weight_decay": 0.01,
    "logging_steps": 10,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    # Mantem as 3 epocas: com 799 exemplos o overfitting comeca cedo e o
    # melhor checkpoint costuma nao ser o ultimo.
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "optim": "paged_adamw_8bit",
    "max_grad_norm": 0.3,
    "seed": RANDOM_SEED,
    "report_to": "none",
}

MAX_SEQ_LENGTH = 2048

# --------------------------------------------------------------------
# Geracao / avaliacao
# --------------------------------------------------------------------
GENERATION_ARGS = {
    "max_new_tokens": 160,
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "repetition_penalty": 1.05,
}

EVAL_RESULTS_FILE = DOCS_DIR / "evaluation_results.json"

# --------------------------------------------------------------------
# Base estruturada de pacientes (dados sinteticos)
# --------------------------------------------------------------------
PATIENTS_DB = DATA_DIR / "patients.db"
PATIENTS_SEED_FILE = DATA_DIR / "patients_seed.json"

# --------------------------------------------------------------------
# Auditoria
# --------------------------------------------------------------------
AUDIT_LOG = LOGS_DIR / "audit.log"

# --------------------------------------------------------------------
# Prompt de instrucao usado no fine-tuning e na inferencia
# --------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Voce e um assistente medico de apoio a decisao clinica. "
    "Responda com base exclusivamente na evidencia fornecida. "
    "Nunca prescreva medicamentos ou dosagens diretamente: toda "
    "sugestao exige validacao de um profissional de saude."
)

INSTRUCTION_TEMPLATE = (
    "Pergunta clinica: {question}\n\n"
    "Evidencia cientifica:\n{context}\n\n"
    "Responda com um veredito (yes/no/maybe) e uma justificativa."
)

# Prefixos da resposta estruturada. Usados no treino e no parser da avaliacao.
VERDICT_PREFIX = "Veredito:"
RATIONALE_PREFIX = "Justificativa:"
