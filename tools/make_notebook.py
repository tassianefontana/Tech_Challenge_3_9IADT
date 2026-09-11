"""Gera notebooks/finetuning_colab.ipynb a partir de celulas em texto."""

import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "finetuning_colab.ipynb"

CELLS = []


def md(src: str):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip().splitlines(keepends=True)})


def code(src: str):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip().splitlines(keepends=True)})


# Reaplicado nas celulas que importam `src`. O %cd altera o diretorio de
# trabalho, mas nao o sys.path: o kernel do Colab inicia em /content e e esse
# caminho que fica registrado. Sem isto, `import src.config` falha.
BOOTSTRAP = """import os, sys

PROJECT_DIR = "/content/techchallenge3"
os.chdir(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)"""


md("""
# Tech Challenge Fase 3 - Fine-tuning QLoRA (PubMedQA)

Assistente medico de apoio a decisao clinica.

| | |
|---|---|
| Modelo base | `Qwen/Qwen2.5-1.5B-Instruct` |
| Tecnica | QLoRA (quantizacao NF4 em 4-bit + adaptadores LoRA) |
| Dataset | PubMedQA PQA-L, curado e anonimizado na Etapa 2 |
| Hardware | GPU T4 16 GB (Colab gratuito) |

> **Antes de comecar:** `Ambiente de execucao > Alterar o tipo de ambiente de execucao > GPU (T4)`
""")

md("## 1. Verificar a GPU alocada")
code("!nvidia-smi")

md("""
## 2. Clonar o repositorio

A URL e lida da variavel de ambiente `REPO_URL`. Defina-a na celula abaixo
ou via `Secrets` do Colab (icone de chave na barra lateral).
""")
code("""
import os

REPO_URL = os.environ.get("REPO_URL", "")

if not REPO_URL:
    try:
        from google.colab import userdata
        REPO_URL = userdata.get("REPO_URL")
    except Exception:
        raise SystemExit(
            "Defina REPO_URL nos Secrets do Colab ou com "
            "os.environ['REPO_URL'] = '...' antes de rodar esta celula."
        )

if not os.path.exists("/content/techchallenge3"):
    !git clone $REPO_URL /content/techchallenge3

%cd /content/techchallenge3
!git pull --ff-only

# Registra o projeto no sys.path para os imports do notebook
import sys
if "/content/techchallenge3" not in sys.path:
    sys.path.insert(0, "/content/techchallenge3")
""")

md("""
## 3. Instalar dependencias

O Colab ja traz `torch` com CUDA, entao instalamos apenas a stack de fine-tuning.

**A celula abaixo reinicia a sessao automaticamente ao final.** Isso e
obrigatorio: o Colab vem com `transformers` 4.x ja carregado em memoria e,
sem reiniciar, o upgrade para a 5.x gera erros do tipo
`ModuleNotFoundError: No module named 'transformers.models.audioflamingo3'`.

Depois do restart, **pule a celula 3 e continue a partir da celula 3.1**.
A celula 2 (`%cd`) precisa ser reexecutada, pois o diretorio de trabalho
volta para `/content`.
""")
code("""
!pip install -q -r requirements-colab.txt

print("Instalacao concluida. Reiniciando a sessao...")
print("Apos o restart: rode a celula 2 (%cd) e siga da celula 3.1.")

import IPython
IPython.Application.instance().kernel.do_shutdown(True)
""")

md("""
### 3.1 Verificar o ambiente apos o restart

Execute esta celula antes de prosseguir. Ela restaura o diretorio de trabalho
e o `sys.path` (perdidos no restart) e valida os imports.
""")
code(BOOTSTRAP + """

import torch, transformers, trl, peft, bitsandbytes
print("cwd         ", os.getcwd())
print("torch       ", torch.__version__, "| cuda:", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("trl         ", trl.__version__)
print("peft        ", peft.__version__)
print("bitsandbytes", bitsandbytes.__version__)

from src.config import BASE_LLM
print("import src  OK -> modelo base:", BASE_LLM)

assert torch.cuda.is_available(), (
    "GPU nao ativa: Ambiente de execucao > Alterar o tipo > T4 GPU"
)
""")

md("""
## 4. Conferir o dataset da Etapa 2

Os arquivos `data/training/*.jsonl` ja vem versionados no repositorio.
Para regenerar do zero (download -> anonimizacao -> curadoria -> split),
descomente as tres primeiras linhas.
""")
code(BOOTSTRAP + """

# !python -m src.preprocessing.download_dataset
# !python -m src.preprocessing.build_dataset
# !python -m src.preprocessing.split_dataset

import json
from src.config import TRAIN_FILE, VAL_FILE, TEST_FILE

for path in (TRAIN_FILE, VAL_FILE, TEST_FILE):
    n = sum(1 for _ in open(path, encoding="utf-8"))
    print(f"{path.name:12s} {n:5d} exemplos")

sample = json.loads(open(TRAIN_FILE, encoding="utf-8").readline())
print("\\nlabel:", sample["label"])
for m in sample["messages"]:
    print("\\n--- " + m["role"] + " ---")
    print(m["content"][:400])
""")

md("""
### 4.1 Distribuicao de tokens

Serve para escolher o `max_length` do treino. Sequencias acima do limite sao
truncadas; muito acima do necessario so desperdicia memoria da GPU.
""")
code(BOOTSTRAP + """

import json
from transformers import AutoTokenizer
from src.config import BASE_LLM, TRAIN_FILE

tok = AutoTokenizer.from_pretrained(BASE_LLM)

lengths = []
with open(TRAIN_FILE, encoding="utf-8") as f:
    for line in f:
        msgs = json.loads(line)["messages"]
        text = tok.apply_chat_template(msgs, tokenize=False)
        lengths.append(len(tok(text)["input_ids"]))

lengths.sort()
n = len(lengths)
print(f"min={lengths[0]}  mediana={lengths[n//2]}  "
      f"p95={lengths[int(n*0.95)]}  max={lengths[-1]}")
print(f"acima de 2048 tokens: {sum(1 for x in lengths if x > 2048)}/{n}")
""")

md("""
## 5. Fine-tuning

Hiperparametros em `src/config.py` (`LORA_CONFIG` e `TRAINING_ARGS`):

- **LoRA r=16, alpha=32** aplicado a atencao (`q,k,v,o`) e MLP (`gate,up,down`)
- **Batch efetivo 16** (batch 2 x grad. accumulation 8) para caber na T4
- **`assistant_only_loss=True`**: a loss e calculada apenas nos tokens da
  resposta, nao no prompt. Sem isso o modelo aprende a reproduzir o abstract.
- **3 epocas** com scheduler cosine e `paged_adamw_8bit`

Tempo medido na T4: **~45 s/step, ~1h50 no total** (150 steps). A lentidao vem
da combinacao dequantizacao 4-bit + gradient checkpointing + T4 sem suporte a
bf16. Nao feche a aba: o Colab gratuito desconecta por inatividade.

Se ocorrer OOM, use `--max-seq-length 1024`. Para encurtar o treino,
`--epochs 2` (~1h15).
""")
code("""
!python -m src.finetuning.train
""")

md("""
### 5.1 Curva de loss

Se a loss de validacao subir enquanto a de treino cai, ha overfitting -
reduza para 2 epocas ou aumente o `lora_dropout`.
""")
code(BOOTSTRAP + """

import json
import matplotlib.pyplot as plt
from src.config import ADAPTER_DIR

state_file = sorted(ADAPTER_DIR.glob("checkpoint-*/trainer_state.json"))[-1]
history = json.load(open(state_file))["log_history"]

train = [(h["epoch"], h["loss"]) for h in history if "loss" in h]
val = [(h["epoch"], h["eval_loss"]) for h in history if "eval_loss" in h]

plt.figure(figsize=(8, 4))
plt.plot(*zip(*train), label="train loss")
if val:
    plt.plot(*zip(*val), marker="o", label="val loss")
plt.xlabel("epoca"); plt.ylabel("loss"); plt.legend(); plt.grid(alpha=.3)
plt.title("Fine-tuning QLoRA - Qwen2.5-1.5B / PubMedQA")
plt.savefig("docs/loss_curve.png", dpi=120, bbox_inches="tight")
plt.show()
""")

md("""
## 6. Avaliacao: base vs fine-tuned

Roda os 102 exemplos do test set nos dois modelos e compara:

- **F1 macro** (metrica principal - o dataset e desbalanceado: 55% yes, 34% no, 11% maybe)
- Acuracia do veredito
- ROUGE-L da justificativa
- Taxa de respostas sem veredito parseavel (o modelo base tende a divagar)

Leva ~10-15 min. Para um teste rapido use `--limit 20`.
""")

code("""
!python -m src.finetuning.evaluate --batch-size 4
""")

md("### 6.1 Matriz de confusao do modelo fine-tuned")
code(BOOTSTRAP + """

import json
from src.config import EVAL_RESULTS_FILE

results = json.load(open(EVAL_RESULTS_FILE, encoding="utf-8"))

for name in ("base", "finetuned"):
    if name not in results:
        continue
    print(f"\\n=== {name.upper()} ===")
    cm = results[name]["matriz_confusao"]
    cols = list(next(iter(cm.values())).keys())
    print("real\\\\pred".ljust(12) + "".join(c.rjust(10) for c in cols))
    for real, row in cm.items():
        print(real.ljust(12) + "".join(str(row[c]).rjust(10) for c in cols))
    print("\\nExemplo de geracao:")
    print(results[name]["exemplos"][0][:300])
""")

md("""
## 7. Teste manual com uma pergunta clinica
""")
code(BOOTSTRAP + """

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.config import BASE_LLM, ADAPTER_DIR, SYSTEM_PROMPT, INSTRUCTION_TEMPLATE

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16)
tok = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))
tok.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(BASE_LLM, quantization_config=bnb,
                                             device_map="auto")
model = PeftModel.from_pretrained(model, str(ADAPTER_DIR)).eval()


def ask(question, context):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INSTRUCTION_TEMPLATE.format(
            question=question, context=context)},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=160, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:],
                      skip_special_tokens=True)


print(ask(
    "Does early mobilization reduce ICU length of stay in mechanically "
    "ventilated patients?",
    "METHODS: Randomized controlled trial with 300 mechanically ventilated "
    "adults comparing early physical therapy versus standard care. "
    "RESULTS: Median ICU stay was 5.9 days in the intervention group versus "
    "7.9 days in controls (p=0.02), with no increase in adverse events.",
))
""")

md("""
## 8. Exportar o adaptador

> **Rode a celula 6 (avaliacao) antes desta.** A limpeza abaixo apaga os
> checkpoints intermediarios; sem eles nao e possivel comparar as epocas.

O adaptador LoRA tem apenas ~30-70 MB (contra ~3 GB do modelo completo),
por isso versionamos so ele. Baixe o `.zip` e descompacte em `models/`
na sua maquina para rodar as Etapas 4 e 5 localmente.
""")
code(BOOTSTRAP + """

from src.config import ADAPTER_DIR

# Remove checkpoints intermediarios para reduzir o tamanho
!rm -rf {ADAPTER_DIR}/checkpoint-*

!du -sh {ADAPTER_DIR}
!cd models && zip -qr /content/adapter.zip $(basename {ADAPTER_DIR})

from google.colab import files
files.download("/content/adapter.zip")
""")

md("""
### 8.1 (Opcional) Publicar no HuggingFace Hub

Alternativa ao download manual: hospedar o adaptador no Hub, o que facilita
carrega-lo no assistente sem versionar binarios no Git.
""")
code("""
# from huggingface_hub import notebook_login
# notebook_login()
#
# repo_id = os.environ["HF_REPO_ID"]
# model.push_to_hub(repo_id)
# tok.push_to_hub(repo_id)
""")

md("""
---
**Proximos passos (fora do Colab):** Etapa 4 (pipeline LangChain + RAG +
base estruturada de pacientes) e Etapa 5 (fluxo LangGraph).
""")

notebook = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

NB.parent.mkdir(parents=True, exist_ok=True)
with open(NB, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"[ok] {NB} ({len(CELLS)} celulas)")
