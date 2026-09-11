# Assistente Médico com LLM Fine-tunada — Tech Challenge Fase 3

Assistente virtual de apoio à decisão clínica construído sobre uma LLM
fine-tunada com QLoRA no dataset **PubMedQA**, orquestrado com **LangChain**
e **LangGraph**, com camada de segurança, auditoria e explainability.

> ⚠️ **Aviso**: sistema acadêmico de apoio à decisão. Não substitui julgamento
> clínico e nunca emite prescrições sem validação humana.

---

## Arquitetura

```
data/raw/          PubMedQA bruto (não versionado, baixado por script)
data/processed/    dataset curado + estatísticas de preprocessing
data/training/     train/val/test em JSONL (formato de chat)
src/config.py      configuração central (paths, modelos, hiperparâmetros)
src/preprocessing/ download, anonimização, curadoria e split
src/finetuning/    treino QLoRA e avaliação base vs fine-tuned
src/rag/           banco vetorial Chroma e retriever
src/app/           pipeline LangChain e interface Streamlit
src/db/            base estruturada de pacientes (SQLite, dados sintéticos)
src/workflow/      fluxo LangGraph
src/security/      guardrails e logging de auditoria
notebooks/         notebooks do Colab (fine-tuning e demonstração)
docs/              relatório técnico, diagrama do fluxo e resultados
```

---

## Instalação

Requer **Python 3.11** (não use 3.13+: `torch` e `bitsandbytes` ainda não têm
wheels estáveis).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS: `source .venv/bin/activate`

---

## Etapa 1 — Preparação dos dados

```bash
python -m src.preprocessing.download_dataset
python -m src.preprocessing.build_dataset
python -m src.preprocessing.split_dataset
```

| Comando | Saída |
|---|---|
| `download_dataset` | `data/raw/ori_pqal.json` (idempotente; use `--force` para rebaixar) |
| `build_dataset` | `data/processed/pubmedqa_processed.json` + `preprocessing_stats.json` |
| `split_dataset` | `data/training/{train,val,test}.jsonl` |

O `build_dataset` aplica, nesta ordem: normalização Unicode NFKC → remoção de
caracteres de controle → **anonimização** (e-mail, URL, CPF, CNS, telefone,
MRN/prontuário, datas, nomes com título clínico, idade ≥ 90 conforme HIPAA
Safe Harbor) → curadoria por tamanho e validade do rótulo → deduplicação
por SHA-256.

O split é **estratificado por rótulo**, com seed fixa (`RANDOM_SEED = 42`),
preservando a distribuição `yes`/`no`/`maybe` nos três conjuntos.

**Resultado:** 1000 exemplos curados (yes 552 / no 338 / maybe 110),
divididos em 799 treino / 99 validação / 102 teste.

---

## Etapa 2 — Fine-tuning (QLoRA)

O treino exige GPU NVIDIA. **Sem GPU local, use o Google Colab:**

1. Faça push deste repositório para o GitHub.
2. Abra `notebooks/finetuning_colab.ipynb` no Colab.
3. `Ambiente de execução > Alterar o tipo > GPU (T4)`.
4. Cadastre a URL do repositório em `Secrets` (ícone de chave) com o nome
   `REPO_URL`.
5. Execute todas as células e baixe o `adapter.zip` ao final.
6. Descompacte em `models/`.

Com GPU local, o mesmo pipeline roda via CLI:

```bash
python -m src.finetuning.train
python -m src.finetuning.train --epochs 2 --max-seq-length 1024   # se houver OOM
```

| Parâmetro | Valor |
|---|---|
| Modelo base | `Qwen/Qwen2.5-1.5B-Instruct` |
| Quantização | NF4 4-bit com double quantization |
| LoRA | r=16, alpha=32, dropout=0.05, em atenção + MLP |
| Batch efetivo | 16 (batch 2 × grad. accumulation 8) |
| Otimizador | `paged_adamw_8bit`, LR 2e-4, scheduler cosine |
| Épocas | 3 |

Tempo medido na T4 do Colab: **~45 s/step, ~1h50 no total** (150 steps).
QLoRA em T4 é lento devido à dequantização 4-bit somada ao gradient
checkpointing e à ausência de suporte a bf16.

A loss é calculada **apenas nos tokens da resposta** (`assistant_only_loss`);
caso contrário o modelo aprenderia a reproduzir o abstract em vez de decidir
o veredito.

---

## Etapa 3 — Avaliação

```bash
python -m src.finetuning.evaluate                 # base + fine-tuned
python -m src.finetuning.evaluate --only base
python -m src.finetuning.evaluate --limit 30      # smoke test
```

Resultados em `docs/evaluation_results.json`.

**F1 macro é a métrica principal**, não a acurácia: o dataset é desbalanceado
(55% `yes`), então um modelo que sempre responde "yes" atinge 55% de acurácia
com F1 macro de apenas ~0.24. Também reportamos ROUGE-L da justificativa e a
taxa de respostas sem veredito parseável.

---

## Etapa 4 — Banco vetorial (RAG)

```bash
python -m src.rag.build_vectordb
python -m src.rag.test_retrieval "Does early mobilization reduce ICU stay?"
```

Chroma + embeddings `all-MiniLM-L6-v2`, persistido em `chromadb/`
(regenerável; não versionado). Cada documento carrega o PMID em metadata,
que é devolvido junto à resposta para garantir **explainability**.

---

## Etapa 5 — Assistente com LangChain

Base estruturada de pacientes (dados **sintéticos**, seed fixa) e pipeline
que combina LLM fine-tunada + evidência recuperada + prontuário:

```bash
python -m src.db.patients            # cria data/patients.db a partir do seed
python -m src.db.patients --force    # recria do zero
```

```python
from src.app.chain import ask

r = ask("Há evidência de benefício da mobilização precoce?", patient_id="PAC-0001")
print(r["answer"])      # resposta já validada pelos guardrails
print(r["sources"])     # PMIDs + prontuário usados (explainability)
print(r["trace_id"])    # correlaciona os eventos em logs/audit.log
```

---

## Etapa 6 — Fluxo automatizado (LangGraph)

O mesmo atendimento como máquina de estados, com desvio condicional,
interrupção precoce e alertas para a equipe médica:

```
triagem ─┬─(bloqueado)──────────────────────────────► END
         └─► prontuário ─► verificar_exames ─┬─(pendentes)─► alerta_exames ─┐
                                             └─(nenhum)───────────────────►┴─► buscar_evidência
buscar_evidência ─► sugerir_conduta ─► guardrail ─► alertar_equipe ─► END
```

```bash
python -m src.workflow.graph --question "Qual a conduta indicada?" --patient PAC-0001
python -m src.workflow.graph --diagram    # exporta docs/fluxo_langgraph.mmd
```

| Nó | Função |
|---|---|
| `triagem` | guardrail de entrada: sanitiza PII, barra prompt injection e temas fora de escopo |
| `carregar_prontuario` | consulta a base estruturada do paciente |
| `verificar_exames` | levanta exames pendentes — define o desvio condicional |
| `alerta_exames` | sinaliza que a conduta foi sugerida com informação incompleta |
| `buscar_evidencia` | recupera os trechos científicos no Chroma |
| `sugerir_conduta` | gera a sugestão com a LLM customizada |
| `guardrail` | valida a saída fora do modelo e anexa o aviso de validação humana |
| `alertar_equipe` | consolida alertas (alergia, tentativa de prescrição, ausência de fonte) |

---

## Etapa 7 — Interface de demonstração

```bash
streamlit run src/app/ui.py
```

Mostra numa única tela o caminho percorrido no grafo, a resposta, os
alertas, as fontes (PMID com link para o PubMed) e a trilha de auditoria
da execução.

Sem GPU local a inferência fica lenta (1–3 min por resposta em CPU). A
alternativa é `notebooks/assistente_demo_colab.ipynb`, que roda o mesmo
código na GPU do Colab e expõe a interface por um túnel temporário.

---

## Testes

```bash
pytest tests/ -q
```

39 testes cobrindo os guardrails (variações de dose, posologia, prompt
injection, vazamento de PII, exigência de citação) e o fluxo LangGraph
(cada caminho do grafo, alertas por severidade, isolamento entre
`trace_id`, correspondência entre o diagrama e os nós implementados).

A LLM e o retriever são substituídos por dublês: os testes verificam
**roteamento e segurança** de forma determinística, sem depender de GPU.

---

## Utilitários

```bash
python tools/make_notebook.py        # regenera o notebook de fine-tuning
python tools/make_demo_notebook.py   # regenera o notebook de demonstração
```

---

## Segurança e conformidade

- Anonimização aplicada antes de qualquer treino ou indexação.
- Guardrails **programáticos** (`src/security/guardrails.py`), fora do modelo:
  o prompt sozinho é contornável. Na entrada, remove PII e barra tentativas de
  sobrescrever as instruções; na saída, redige dose e posologia, detecta
  linguagem prescritiva e exige citação de PMID.
- Toda interação é registrada em `logs/audit.log` (JSONL, não versionado),
  com `trace_id` correlacionando os eventos de uma mesma execução.
- Respostas sempre acompanhadas da fonte (PMID) utilizada e do aviso de
  validação humana obrigatória.


---

## Documentação

- `docs/guia_do_projeto.md` — **guia didático**: explica cada arquivo e cada
  etapa do zero, com glossário. Comece por aqui se o projeto é novo para você
- `docs/relatorio_tecnico.md` — relatório técnico completo
- `docs/evaluation_results.json` — métricas da avaliação
- `data/processed/preprocessing_stats.json` — estatísticas de curadoria
