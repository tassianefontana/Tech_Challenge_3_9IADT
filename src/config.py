"""Configuracao central do projeto.

Todos os modulos devem importar caminhos e constantes daqui, evitando
paths hardcoded divergentes entre scripts (problema que causava o
vectorstore ser gravado em `./chromadb` e lido de `./vectordb`).
"""

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

# Diretorio unico do banco vetorial (fonte da verdade)
VECTORDB_DIR = BASE_DIR / "chromadb"
VECTORDB_COLLECTION = "pubmedqa"

for _d in (RAW_DIR, PROCESSED_DIR, TRAINING_DIR, MODELS_DIR, LOGS_DIR):
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
