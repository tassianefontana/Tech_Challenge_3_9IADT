# Guia do Projeto — explicação completa para quem está começando

Este documento explica **o que cada arquivo faz e por quê**, em ordem de
execução. Não assume conhecimento prévio de LLMs, LangChain ou fine-tuning.

Se você quer só rodar o projeto, use o `README.md`. Este guia é para
**entender**.

---

## Sumário

1. [O problema que estamos resolvendo](#1-o-problema)
2. [Conceitos básicos](#2-conceitos-basicos)
3. [Visão geral da arquitetura](#3-visao-geral)
4. [Etapa 1 — Estrutura do projeto](#4-etapa-1)
5. [Etapa 2 — Preparação dos dados](#5-etapa-2)
6. [Etapa 3 — Fine-tuning](#6-etapa-3)
7. [Etapa 4 — Avaliação](#7-etapa-4)
8. [Etapa 5 — RAG](#8-etapa-5)
9. [Etapa 6 — Assistente com LangChain](#9-etapa-6)
10. [O que ainda falta](#10-o-que-falta)
11. [Glossário](#11-glossario)

---

<a name="1-o-problema"></a>
## 1. O problema que estamos resolvendo

O enunciado pede um **assistente virtual médico** que:

- responda dúvidas clínicas de médicos;
- use os dados do próprio hospital (não só conhecimento genérico da internet);
- consulte informações atualizadas do paciente (prontuário, exames);
- **nunca** prescreva sozinho, sem validação humana;
- registre tudo para auditoria;
- diga **de onde tirou** cada informação.

Um detalhe importante: não temos acesso a dados reais de um hospital.
O enunciado sugere o dataset público **PubMedQA**, que escolhemos. Ele contém
1000 perguntas clínicas com o resumo (abstract) do artigo científico que a
responde, e um veredito: `yes`, `no` ou `maybe`.

Exemplo real do dataset:

> **Pergunta:** As mitocôndrias têm papel na remodelação das folhas da planta
> renda durante a morte celular programada?
> **Evidência:** (abstract com seções BACKGROUND, METHODS, RESULTS...)
> **Veredito:** `yes`
> **Justificativa:** "Results depicted mitochondrial dynamics in vivo as
> PCD progresses within the lace plant..."

Como o PubMedQA não tem prontuários, nós **criamos uma base sintética de
pacientes** para simular os registros internos do hospital. Isso é explicado
na seção da Etapa 6.

---

<a name="2-conceitos-basicos"></a>
## 2. Conceitos básicos

Se você já domina esses termos, pule para a seção 3.

### O que é uma LLM

Uma **LLM** (Large Language Model, ou "modelo de linguagem grande") é um
programa que aprendeu a prever a próxima palavra de um texto. Treinado em
bilhões de frases, ele acaba conseguindo escrever, resumir e responder
perguntas.

Usamos o **Qwen2.5-1.5B-Instruct**. O "1.5B" significa 1,5 bilhão de
parâmetros — os números internos que o modelo ajusta ao aprender. É um modelo
pequeno para os padrões atuais (o GPT-4 tem centenas de bilhões), escolhido
justamente porque **cabe numa GPU gratuita do Google Colab**.

### Token

Modelos não leem letras nem palavras: leem **tokens**, pedaços de palavra.
"Insuficiência" pode virar 3 tokens. Como regra grosseira, 1 token ≈ 0,75
palavra em inglês. Isso importa porque todo modelo tem um **limite de tokens**
que consegue processar de uma vez (no nosso caso, 2048).

### Fine-tuning

O modelo genérico sabe conversar, mas não conhece o estilo do nosso problema.
**Fine-tuning** é continuar o treino com exemplos específicos, para que ele
aprenda o formato e o domínio desejados.

Analogia: contratar alguém formado em medicina (o modelo base) e treiná-lo
nos protocolos específicos do seu hospital (fine-tuning).

### LoRA e QLoRA

Treinar todos os 1,5 bilhão de parâmetros exigiria uma GPU cara.

**LoRA** (Low-Rank Adaptation) é um truque: em vez de alterar o modelo
inteiro, você **congela** o modelo original e treina apenas pequenas matrizes
extras acopladas a ele. No nosso caso, treinamos **18,4 milhões** de
parâmetros — apenas **1,18%** do total. O resultado é um "adaptador" de
poucas dezenas de MB, em vez de um modelo novo de 3 GB.

**QLoRA** = LoRA + **quantização**. Quantizar é guardar os números do modelo
com menos precisão (4 bits em vez de 16), como salvar uma foto em qualidade
menor. O modelo ocupa ~4x menos memória e perde pouquíssima qualidade. É o
que permite treinar numa GPU T4 de 16 GB.

### RAG

**RAG** (Retrieval-Augmented Generation, "geração aumentada por recuperação")
resolve um problema sério: LLMs **inventam informação** com confiança
(fenômeno chamado *alucinação*). Inaceitável em medicina.

A ideia do RAG:

1. O médico faz uma pergunta.
2. Antes de responder, o sistema **busca** os documentos mais relevantes numa
   base confiável.
3. Esses documentos são colados no prompt.
4. A LLM responde **usando aquele texto**, e cita a fonte.

Assim a resposta fica ancorada em evidência real e rastreável.

### Embeddings e banco vetorial

Como o computador "busca por significado" e não por palavra exata?

Um modelo de **embedding** transforma um texto num vetor — uma lista de
números (no nosso caso, 384). Textos com sentido parecido geram vetores
próximos no espaço. "infarto do miocárdio" e "ataque cardíaco" ficam perto,
mesmo sem compartilhar nenhuma palavra.

Um **banco vetorial** (usamos o **Chroma**) guarda esses vetores e responde
rápido à pergunta "quais os 3 textos mais parecidos com este?".

### LangChain e LangGraph

**LangChain** é uma biblioteca para montar pipelines com LLMs: conectar
prompt → modelo → parser → busca, sem escrever tudo na mão.

**LangGraph** é para fluxos com **decisões e ramificações**. Ex.: "se o
paciente tem exame pendente, vá para o nó de alerta; senão, siga para a
sugestão de conduta". É um grafo de etapas, não uma linha reta.

---

<a name="3-visao-geral"></a>
## 3. Visão geral da arquitetura

```
        ETAPA 2                ETAPA 3              ETAPA 4
   ┌──────────────┐      ┌──────────────┐     ┌──────────────┐
   │  PubMedQA    │      │  Fine-tuning │     │  Avaliação   │
   │  bruto       │─────▶│    QLoRA     │────▶│ base vs FT   │
   │  (1000 QAs)  │      │   (Colab)    │     │  F1, ROUGE   │
   └──────────────┘      └──────────────┘     └──────────────┘
     limpeza                 adaptador
     anonimização            LoRA (~30MB)
     curadoria                    │
     split                        │
          │                       ▼
          │              ┌──────────────────────────────┐
          │              │      ETAPA 6 — Assistente     │
          └─────────────▶│                              │
       ETAPA 5           │  pergunta do médico          │
   ┌──────────────┐      │        │                     │
   │ Banco        │─────▶│        ├── RAG (evidência)   │
   │ vetorial     │      │        ├── prontuário SQLite │
   │ (Chroma)     │      │        └── LLM fine-tunada   │
   └──────────────┘      │                              │
                         │  resposta + FONTES (PMIDs)   │
   ┌──────────────┐      │                              │
   │ Pacientes    │─────▶│                              │
   │ (SQLite)     │      └──────────────────────────────┘
   └──────────────┘
```

Estrutura de pastas:

```
TechChallenge3/
├── data/
│   ├── raw/            PubMedQA original (não vai pro Git: é baixável)
│   ├── processed/      dataset limpo e anonimizado
│   ├── training/       train/val/test prontos para treino
│   └── patients_seed.json   pacientes sintéticos
├── src/
│   ├── config.py       ⭐ configuração central
│   ├── preprocessing/  Etapa 2
│   ├── finetuning/     Etapas 3 e 4
│   ├── rag/            Etapa 5
│   ├── db/             base de pacientes
│   ├── app/            Etapa 6 (assistente)
│   ├── security/       (a fazer) guardrails e auditoria
│   └── workflow/       (a fazer) fluxo LangGraph
├── notebooks/          fine-tuning no Colab
├── tools/              utilitários de desenvolvimento
└── docs/               relatórios
```

---

<a name="4-etapa-1"></a>
## 4. Etapa 1 — Estrutura do projeto

### `.gitignore`

Lista de arquivos que o Git deve **ignorar**. Regra geral: não versione o que
é (a) gerável por script, (b) gigante, ou (c) secreto.

| Ignorado | Por quê |
|---|---|
| `.venv/` | milhares de arquivos de bibliotecas; cada um instala as suas |
| `data/raw/` | baixável com um comando |
| `chromadb/` | banco vetorial regenerável (32 MB) |
| `models/` | pesos de modelo, centenas de MB |
| `data/patients.db` | regenerável a partir do seed em JSON |
| `logs/` | logs de auditoria, podem conter dados sensíveis |
| `.env` | senhas e tokens — **nunca** commitar |

### `requirements.txt` e `requirements-colab.txt`

Listas de bibliotecas com **versão fixa** (`langchain==1.4.0`). Fixar versão
evita que o projeto quebre quando uma biblioteca lançar uma atualização
incompatível.

São dois arquivos porque os ambientes diferem: o Colab já vem com `torch` +
CUDA instalado, e reinstalar quebraria a GPU. O arquivo do Colab traz apenas
a stack de treino (`trl`, `peft`, `bitsandbytes`).

> **Lição prática deste projeto:** o `requirements.txt` inicial foi escrito
> "de cabeça" com versões antigas (langchain 0.3, transformers 4.x). Ao
> conferir o ambiente real, ele tinha langchain 1.4 e transformers 5.17 —
> APIs bem diferentes. Sempre confira o que está de fato instalado antes de
> escrever código: `pip list`.

### `src/config.py` ⭐

**O arquivo mais importante da organização.** Centraliza todos os caminhos,
nomes de modelo e hiperparâmetros. Nenhum outro arquivo escreve um caminho
"na mão".

```python
BASE_DIR = Path(__file__).resolve().parent.parent
VECTORDB_DIR = BASE_DIR / "chromadb"
```

`Path(__file__)` é o caminho deste próprio arquivo; `.parent.parent` sobe dois
níveis até a raiz do projeto. Assim tudo funciona independentemente de onde
você executa o script, no Windows ou no Linux.

**Por que isso importa (caso real):** antes do `config.py`, um script gravava
o banco vetorial em `./chromadb` e outro lia de `./vectordb`. Eram pastas
diferentes — as buscas retornavam vazio silenciosamente, sem erro nenhum.
Esse tipo de bug custa horas. Com a configuração central, é impossível.

O arquivo também define os prompts:

```python
SYSTEM_PROMPT = (
    "Voce e um assistente medico de apoio a decisao clinica. "
    "Responda com base exclusivamente na evidencia fornecida. "
    "Nunca prescreva medicamentos ou dosagens diretamente..."
)
```

O **system prompt** define a "persona" e as regras do modelo. Ele é usado
tanto no treino quanto na inferência — e precisa ser **o mesmo nos dois**,
senão o modelo se comporta de forma inconsistente.

### Os arquivos `__init__.py`

Arquivos vazios que transformam pastas em **pacotes Python**. Sem eles,
`from src.config import ...` não funciona.

> **Detalhe que gerou uma correção:** a pasta original chamava-se
> `src/langgraph`, igual à biblioteca `langgraph`. Em certos contextos o
> Python importaria a nossa pasta em vez da biblioteca, gerando erros
> confusos. Renomeamos para `src/workflow`. **Nunca dê a um módulo seu o nome
> de uma biblioteca que você usa.**

---

<a name="5-etapa-2"></a>
## 5. Etapa 2 — Preparação dos dados

Etapa menos glamourosa e mais decisiva: modelo treinado com dado ruim aprende
coisa ruim.

### `src/preprocessing/download_dataset.py`

Baixa o `ori_pqal.json` do repositório oficial do PubMedQA.

É **idempotente**: se o arquivo já existe, não baixa de novo (use `--force`
para forçar). Scripts idempotentes podem ser executados várias vezes sem
efeito colateral — importante em pipelines.

### `src/preprocessing/anonymizer.py`

Remove informação que identifica pessoas (**PII** — Personally Identifiable
Information), substituindo por marcadores:

| Padrão | Vira |
|---|---|
| `joao@hospital.com` | `[EMAIL]` |
| `123.456.789-00` | `[CPF]` |
| `(11) 98765-4321` | `[PHONE]` |
| `prontuário 12345` | `[MRN]` |
| `12/03/2024` | `[DATE]` |
| `Dr. João Silva` | `[NAME]` |
| `94 years old` | `[AGE_90+]` |

Funciona com **expressões regulares** (regex) — padrões de busca em texto.
Exemplo, o de e-mail:

```python
re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
```

Lê-se: um ou mais caracteres de palavra, um `@`, um domínio, um ponto, uma
extensão.

**Duas decisões que valem entender:**

**Por que substituir em vez de apagar?** Se apagássemos, "paciente atendido em
12/03/2024" viraria "paciente atendido em", e o modelo perderia a noção de
que ali havia uma data. Com `[DATE]`, a estrutura da frase se mantém. Isso se
chama **pseudonimização**.

**Por que idade ≥ 90 é PII?** Pouquíssimas pessoas têm mais de 90 anos. Uma
idade de 97 anos, combinada com cidade e data de internação, pode identificar
alguém. A norma americana **HIPAA Safe Harbor** exige agrupar todas as idades
acima de 89. É um **identificador indireto** — o mesmo raciocínio vale para
CEP completo e datas exatas.

**"Mas o PubMedQA é público, por que anonimizar?"** Porque o enunciado exige
a técnica, e porque o pipeline foi feito para funcionar com prontuários
reais. A camada está pronta: trocando a fonte de dados, a conformidade com
LGPD/HIPAA continua valendo. Rodando sobre o PubMedQA, ela encontrou 10 datas
e 1 URL — pouco, como esperado, mas comprova que funciona.

A função `contains_pii()` será reaproveitada nos guardrails, para bloquear um
médico que cole dados identificáveis na pergunta.

### `src/preprocessing/build_dataset.py`

Pipeline principal, em 5 fases:

**1. Limpeza**

```python
text = unicodedata.normalize("NFKC", text)
```

Unicode permite escrever o mesmo caractere de várias formas (o "é" pode ser
um caractere só ou "e" + acento separado). Para o computador são strings
diferentes. `NFKC` padroniza tudo numa forma só. Também removemos caracteres
de controle invisíveis e espaços múltiplos.

**2. Preservação da estrutura do abstract**

O PubMedQA traz o abstract em pedaços (`CONTEXTS`) com rótulos (`LABELS`):
BACKGROUND, METHODS, RESULTS, CONCLUSIONS. Nós mantemos os rótulos:

```
BACKGROUND: Programmed cell death is the regulated death of cells...
METHODS: Mitochondrial dynamics were examined...
RESULTS: Results depicted...
```

**Por quê:** "RESULTS" e "METHODS" têm pesos muito diferentes numa conclusão
clínica. Jogar tudo num blocão faria o modelo perder essa distinção. Um
detalhe pequeno com impacto real na qualidade.

**3. Anonimização** — chama o módulo anterior nos três campos.

**4. Curadoria**

Descarta ou ajusta exemplos ruins:

- sem pergunta → descarta
- veredito fora de `yes/no/maybe` → descarta
- resposta com menos de 20 caracteres → descarta (não ensina nada)
- contexto com menos de 100 caracteres → descarta
- contexto acima de 6000 caracteres → **trunca** na última palavra inteira

**5. Deduplicação**

```python
key = f"{question.lower()}||{answer.lower()}"
return hashlib.sha256(key.encode("utf-8")).hexdigest()
```

Gera uma "impressão digital" (hash) de cada exemplo. Duplicatas produzem o
mesmo hash e a segunda é descartada. Duplicatas são perigosas: o modelo vê o
mesmo exemplo várias vezes e o memoriza; pior, se a mesma pergunta cair no
treino **e** no teste, a avaliação fica inflada (*data leakage*).

**Resultado real:** 1000 exemplos entraram, 1000 saíram — nenhum descarte.
Faz sentido: o PQA-L é o subconjunto **rotulado manualmente por especialistas**
do PubMedQA, ou seja, já vem curado. Os filtros não são inúteis: eles provam
a qualidade e protegeriam o pipeline com dados sujos de verdade.

O script gera também `preprocessing_stats.json`, com as contagens — evidência
objetiva para o relatório técnico.

### `src/preprocessing/split_dataset.py`

Divide os dados em três conjuntos:

| Conjunto | Tamanho | Para quê |
|---|---|---|
| **train** | 799 | o modelo aprende com estes |
| **val** | 99 | acompanhar o treino e detectar overfitting |
| **test** | 102 | avaliação final — o modelo **nunca** os vê no treino |

Separar teste é inegociável: avaliar com dados de treino é como aplicar uma
prova cujas respostas o aluno decorou.

**Split estratificado.** Nosso dataset é desbalanceado: 552 `yes`, 338 `no`,
110 `maybe`. Num sorteio aleatório simples, o teste poderia sair com poucos
`maybe` (ou nenhum), e a avaliação dessa classe viraria ruído. Estratificar
significa sortear **dentro de cada classe** e depois juntar, preservando a
proporção:

```
train  799 (yes=441, no=270, maybe=88)
val     99 (yes=55,  no=33,  maybe=11)
test   102 (yes=56,  no=35,  maybe=11)
```

**Seed fixa.** `random.Random(42)` faz o sorteio ser sempre o mesmo. Sem isso,
cada execução produziria uma divisão diferente e você não conseguiria comparar
dois treinos — nem saber se a melhora veio do modelo ou da sorte no sorteio.
Isso se chama **reprodutibilidade**.

**Formato de saída (JSONL).** Um JSON por linha, permitindo ler arquivos
enormes sem carregar tudo na memória. Cada linha:

```json
{"id": "23972333", "label": "yes", "messages": [
  {"role": "system",    "content": "Voce e um assistente medico..."},
  {"role": "user",      "content": "Pergunta clinica: ...\n\nEvidencia: ..."},
  {"role": "assistant", "content": "Veredito: yes\nJustificativa: ..."}
]}
```

Esse é o **formato conversacional**: system (as regras), user (o que o médico
pergunta), assistant (a resposta ideal). É exatamente o que a biblioteca de
treino espera, sem conversão adicional.

Repare que a resposta tem formato rígido — `Veredito: X` + `Justificativa: Y`.
Isso é proposital: facilita extrair a decisão automaticamente na avaliação, e
ensina o modelo a ser consistente.

---

<a name="6-etapa-3"></a>
## 6. Etapa 3 — Fine-tuning

### `src/finetuning/train.py`

O coração do treino. Vamos por partes.

**Carregar o modelo em 4 bits:**

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

- `load_in_4bit`: cada número ocupa 4 bits em vez de 16. ~4x menos memória.
- `nf4`: formato de quantização desenhado para pesos de redes neurais
  (que seguem distribuição normal). Perde menos qualidade que o 4-bit comum.
- `double_quant`: quantiza também as constantes da quantização. Economia extra.
- `compute_dtype`: as **contas** são feitas em 16 bits. Só o armazenamento é
  4 bits.

**Acoplar o LoRA:**

```python
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
```

`gradient_checkpointing` é uma troca de memória por tempo: em vez de guardar
todos os resultados intermediários, o modelo recalcula alguns quando precisa.
Fica ~20% mais lento, mas cabe na GPU.

Configuração do LoRA:

```python
"r": 16,             # "espessura" das matrizes extras
"lora_alpha": 32,    # o quanto elas influenciam (convenção: 2x o r)
"lora_dropout": 0.05,# desliga 5% das conexões aleatoriamente (anti-overfitting)
"target_modules": ["q_proj","k_proj","v_proj","o_proj",
                   "gate_proj","up_proj","down_proj"],
```

`target_modules` diz **onde** acoplar os adaptadores. Os quatro primeiros são
o mecanismo de **atenção** (como o modelo decide o que é relevante); os três
últimos são a **MLP** (onde fica boa parte do conhecimento). Muitos tutoriais
aplicam só na atenção; incluir a MLP costuma render resultado melhor.

Saída real do treino:

```
trainable params: 18,464,768 || all params: 1,562,179,072 || trainable%: 1.1820
```

Treinamos 1,18% do modelo. É esse o ganho do LoRA.

**Configurar o treino:**

```python
"num_train_epochs": 3,
"per_device_train_batch_size": 2,
"gradient_accumulation_steps": 8,
"learning_rate": 2e-4,
"lr_scheduler_type": "cosine",
"warmup_steps": 5,
"optim": "paged_adamw_8bit",
```

- **Época**: uma passada completa por todos os 799 exemplos. Fazemos 3.
- **Batch**: quantos exemplos o modelo vê antes de ajustar os pesos. Só 2
  cabem na T4.
- **Gradient accumulation**: acumula os gradientes de 8 batches antes de
  atualizar. Efeito equivalente a batch 16, sem gastar mais memória.
  Batch maior = aprendizado mais estável.
- **Learning rate** (2e-4 = 0,0002): o tamanho do passo. Alto demais, o modelo
  "pula" a solução; baixo demais, não aprende. `2e-4` é o padrão para LoRA —
  bem maior que o de fine-tuning completo (~2e-5), porque há poucos parâmetros.
- **Scheduler cosine**: a taxa começa alta e diminui suavemente até quase
  zero, seguindo uma curva de cosseno. Aprende rápido no início e refina no
  fim.
- **Warmup**: os primeiros 5 steps usam taxa crescente a partir do zero.
  Evita que atualizações bruscas no início desestabilizem o modelo.
- **paged_adamw_8bit**: otimizador que guarda seu estado em 8 bits e usa
  memória paginada, evitando estouro de VRAM.

Total: 799 ÷ 16 × 3 ≈ **150 steps**.

**O parâmetro mais importante:**

```python
assistant_only_loss=True
```

A **loss** (perda) mede o erro do modelo. Por padrão, ela é calculada sobre
**todos** os tokens — incluindo a pergunta e o abstract de 1400 caracteres
que nós mesmos fornecemos.

Isso é ruim: o modelo gastaria a maior parte do esforço aprendendo a
**reproduzir o abstract**, e quase nada aprendendo a **decidir o veredito**.
Com `assistant_only_loss=True`, a loss considera apenas os tokens da resposta.
Ele aprende a tarefa certa.

**Erro amigável se não houver GPU:**

```python
if not torch.cuda.is_available():
    raise SystemExit("GPU NVIDIA nao detectada... Use o Google Colab...")
```

Melhor que um erro críptico do CUDA 200 linhas depois.

### `notebooks/finetuning_colab.ipynb`

Como treinar sem GPU: o Google Colab oferece uma **Tesla T4 (16 GB)** de
graça. O notebook clona o repositório, instala as dependências e roda o mesmo
`train.py` — nada é duplicado.

**Três armadilhas do Colab que o notebook já resolve** (todas encontradas na
prática):

**1. Reiniciar após instalar.** O Colab já vem com `transformers` 4.x
carregado na memória. Ao instalar a 5.x, os arquivos em disco mudam mas o
Python continua usando o que já carregou — resultado:
`ModuleNotFoundError: No module named 'transformers.models.audioflamingo3'`.
Solução: a célula de instalação reinicia a sessão automaticamente.

**2. `%cd` não muda o `sys.path`.** O kernel inicia em `/content` e é esse
caminho que fica registrado para imports. Mudar de pasta com `%cd` não altera
isso, então `import src.config` falha com `No module named 'src'`. Solução:
cada célula que importa `src` começa registrando o caminho explicitamente.
Bônus: cada célula virou independente, então dá para retomar de qualquer
ponto após um crash.

**3. Bibliotecas mudam de API.** O `warmup_ratio` que usávamos foi removido
no trl 1.x, substituído por `warmup_steps` — `TypeError` no meio do treino.
Lição: quando um parâmetro der erro, **verifique todos os outros de uma vez**
em vez de corrigir um por um.

### `tools/make_notebook.py`

Notebook `.ipynb` é um JSON complicado, horrível de editar à mão e de revisar
em diff. Este script gera o notebook a partir de texto Python simples. Para
alterar o notebook, edite este arquivo e rode `python tools/make_notebook.py`.

---

<a name="7-etapa-4"></a>
## 7. Etapa 4 — Avaliação

### `src/finetuning/evaluate.py`

Roda os 102 exemplos de teste no modelo **base** e no **fine-tunado**, e
compara. Sem essa comparação, não há como afirmar que o fine-tuning serviu
para alguma coisa.

**Acurácia não basta.** Acurácia é a fração de acertos. Parece a métrica
óbvia, mas com dados desbalanceados ela engana:

> 55% dos exemplos são `yes`. Um modelo burro que responde **sempre "yes"**
> acerta 55%. Parece razoável — e é inútil.

**F1 macro** resolve isso. Para cada classe calcula-se:

- **Precisão**: dos que chamei de `yes`, quantos eram mesmo?
- **Recall**: dos `yes` que existiam, quantos eu encontrei?
- **F1**: a média harmônica das duas.

O **macro** tira a média simples dos F1 das três classes, dando a `maybe`
(110 exemplos) o mesmo peso de `yes` (552). O modelo "sempre yes" tira F1
macro de ~0,24 — expondo a inutilidade que a acurácia escondia.

**ROUGE-L** mede a qualidade da justificativa em texto livre, procurando a
maior subsequência comum entre a resposta gerada e a de referência.
Implementamos com programação dinâmica em duas linhas de memória (`O(min(n,m))`).
Validado: `"the cat sat on the mat"` vs `"the cat is on the mat"` = 0.8333.

**Taxa de não-parseadas**: quantas respostas não tinham um veredito
identificável. Espera-se que o modelo base divague ("Well, it depends on...")
e o fine-tunado siga o formato. Essa métrica sozinha costuma justificar o
fine-tuning.

**Matriz de confusão**: mostra *quais* erros o modelo comete — por exemplo,
se ele confunde `maybe` com `yes` sistematicamente. Muito mais informativo que
um número único.

**Um detalhe técnico que quebra tudo silenciosamente:**

```python
tokenizer.padding_side = "left"   # geração
tokenizer.padding_side = "right"  # treino
```

Para processar vários exemplos juntos, todos precisam do mesmo comprimento —
completa-se com *padding*. No **treino**, o padding vai à direita. Na
**geração**, precisa ir à esquerda: o modelo continua o texto a partir do
último token, e se houver padding no fim, ele tenta continuar a partir do
vazio, produzindo lixo. Erro clássico, e não gera mensagem de erro nenhuma.

---

<a name="8-etapa-5"></a>
## 8. Etapa 5 — RAG (banco vetorial)

### `src/rag/vectorstore.py`

Ponto **único** de acesso ao banco vetorial. Todo mundo pega a conexão daqui.

```python
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
```

`@lru_cache` faz a função ser executada **uma vez só**; nas chamadas
seguintes devolve o resultado guardado. Sem isso, o modelo de embeddings
(90 MB) seria recarregado a cada busca.

### `src/rag/build_vectordb.py`

Transforma os 1000 exemplos em vetores e grava no Chroma.

Cada documento vira um texto com pergunta + evidência + conclusão, mais os
**metadados**:

```python
metadata={
    "pmid": item["id"],
    "label": item["label"],
    "source": f"PubMed PMID {item['id']}",
    "question": item["question"],
}
```

Os metadados viajam junto com o vetor. É por causa deles que conseguimos dizer
*de onde veio* cada informação — a base da **explainability** exigida pelo
enunciado.

Usamos o embedding **all-MiniLM-L6-v2**: pequeno (90 MB), rápido, roda em CPU
e tem qualidade suficiente.

### `src/rag/test_retrieval.py`

Testa a busca. Resultado real:

```
Consulta: Do mitochondria play a role in remodelling lace plant leaves...
[1] PMID 21645374   ← exatamente o artigo certo, em primeiro lugar
```

**Limitação importante e honesta.** Testando "Does early mobilization reduce
ICU length of stay?", o sistema retornou artigos sobre infarto e qualidade
hospitalar em fins de semana — irrelevantes.

Isso **não é um bug**. O PQA-L tem apenas 1000 abstracts e simplesmente não
cobre a maioria dos temas clínicos. O buscador faz seu trabalho: devolve o
mais próximo que existe. Se não existe nada próximo, o resultado é ruim.

Duas consequências práticas:
1. Na demonstração em vídeo, use perguntas cujos artigos existam no corpus.
2. Isso deve constar como limitação no relatório técnico. Num hospital real,
   a base teria centenas de milhares de documentos.

---

<a name="9-etapa-6"></a>
## 9. Etapa 6 — Assistente com LangChain

### `src/db/patients.py`

O PubMedQA tem evidência científica, mas o enunciado pede consultas a "bases
de dados estruturadas (prontuários e registros)". Criamos uma base **SQLite**
com pacientes **sintéticos**.

SQLite é um banco de dados que vive num único arquivo, sem servidor — já vem
com o Python. Ideal aqui.

Quatro tabelas relacionadas:

```
pacientes ──┬── exames      (nome, status pendente/concluído, resultado)
            ├── medicacoes  (nome, posologia, ativo)
            └── alergias    (substância)
```

São 20 pacientes gerados com seed fixa (portanto sempre iguais), com
diagnósticos plausíveis e ~35% dos exames marcados como pendentes — para que
o fluxo de decisão da próxima etapa tenha o que detectar.

Exemplo de saída:

```
Paciente PAC-0003 (J.U.), 67 anos, sexo F
Setor: Enfermaria | Internado em: 2025-05-26
Diagnostico: Insuficiencia cardiaca descompensada
Alergias: Penicilina, Latex
Exames concluidos: BNP (dentro da normalidade); Troponina (normalidade)
Exames PENDENTES: Ecocardiograma
Medicacoes ativas: Insulina regular conforme glicemia capilar
```

**Nenhum dado real é usado** — os pacientes são fictícios, identificados por
iniciais falsas. O `.db` fica fora do Git; versionamos o `patients_seed.json`,
a partir do qual ele é reconstruído. (Dado gerável não se versiona.)

A função `format_patient()` transforma o registro em texto para injetar no
prompt, e `get_pending_exams()` será usada pelo nó de verificação do fluxo
LangGraph.

### `src/app/llm.py`

Carrega a LLM fine-tunada e a entrega como componente LangChain.

Detecta o ambiente automaticamente: com GPU, carrega em 4 bits; sem GPU, em
float32 (lento, só para testes). Se o adaptador ainda não existir, usa o
modelo base e **avisa** — foi isso que permitiu construir todo o assistente
enquanto o fine-tuning ainda rodava no Colab.

```python
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
```

`merge_and_unload()` funde os pesos do LoRA no modelo base. Depois disso não
há mais "adaptador separado": vira um modelo só, e a inferência fica mais
rápida.

### `src/app/chain.py`

Junta tudo. A função `ask(question, patient_id)` faz:

1. Busca o prontuário no SQLite (se um paciente foi informado).
2. Busca os 3 documentos mais relevantes no banco vetorial.
3. Monta o prompt com pergunta + prontuário + evidência.
4. Chama a LLM fine-tunada.
5. Devolve resposta **e** lista de fontes.

O prompt carrega regras explícitas:

```
1. Baseie-se exclusivamente na evidencia e no prontuario fornecidos.
2. Se a informacao for insuficiente, declare isso explicitamente.
3. Cite o PMID da evidencia que sustenta cada afirmacao.
4. Nunca indique dose ou posologia; encaminhe para validacao medica.
```

A regra 2 combate alucinação; a 3 força explainability; a 4 é o guardrail de
segurança no nível da instrução.

A evidência é numerada e prefixada com o PMID, para que o modelo consiga
citar:

```
[1] PMID 21645374
Pergunta: ...
Evidencia: ...
```

E o retorno inclui as fontes de forma estruturada:

```json
{
  "answer": "...",
  "sources": [
    {"tipo": "evidencia_cientifica", "pmid": "21645374",
     "url": "https://pubmed.ncbi.nlm.nih.gov/21645374/", "conclusao": "yes"},
    {"tipo": "prontuario", "paciente_id": "PAC-0003"}
  ]
}
```

O médico pode **clicar e conferir o artigo original**. Isso é explainability
de verdade — não basta o modelo afirmar que consultou uma fonte.

---

<a name="10-o-que-falta"></a>
## 10. O que ainda falta

| Etapa | Status |
|---|---|
| Estrutura do projeto | Concluída |
| Preparação dos dados | Concluída |
| Fine-tuning | Código pronto, **treino em execução no Colab** |
| Avaliação | Código pronto, aguarda o adaptador |
| RAG | Concluída |
| Assistente LangChain | Concluída (falta testar com o modelo treinado) |
| **Fluxo LangGraph** (`src/workflow/`) | **A fazer** |
| **Guardrails e auditoria** (`src/security/`) | **A fazer** |
| **Relatório técnico** (`docs/`) | **A fazer** |
| **Vídeo de 15 min** | **A fazer** |

O fluxo LangGraph terá nós como: triagem → buscar prontuário → verificar
exames pendentes → buscar evidência → sugerir conduta → guardrail → alertar
equipe, com desvio condicional quando houver exame pendente.

Os guardrails farão a validação **programática** (hoje as regras existem só
no prompt, e prompt pode ser contornado): bloquear dose/posologia na saída,
filtrar PII na entrada e registrar tudo em log estruturado.

---

<a name="11-glossario"></a>
## 11. Glossário

| Termo | Significado |
|---|---|
| **Abstract** | Resumo de um artigo científico |
| **Adapter** | Arquivo pequeno com os pesos do LoRA treinado |
| **Alucinação** | Quando a LLM inventa informação com aparência convincente |
| **Batch** | Grupo de exemplos processados juntos |
| **Chroma** | Banco de dados vetorial usado no projeto |
| **Curadoria** | Seleção e limpeza dos exemplos do dataset |
| **Data leakage** | Dados de teste vazando para o treino; infla a avaliação |
| **Embedding** | Representação numérica (vetor) do significado de um texto |
| **Época** | Uma passada completa por todo o conjunto de treino |
| **Explainability** | Capacidade de mostrar a origem de cada informação |
| **F1 macro** | Métrica que trata todas as classes com o mesmo peso |
| **Fine-tuning** | Continuar o treino de um modelo com dados específicos |
| **Guardrail** | Regra que limita o que o sistema pode responder |
| **Hiperparâmetro** | Configuração do treino (learning rate, épocas...) |
| **JSONL** | Um objeto JSON por linha do arquivo |
| **LangChain** | Biblioteca para montar pipelines com LLMs |
| **LangGraph** | Biblioteca para fluxos com ramificação e decisão |
| **Learning rate** | Tamanho do passo de ajuste a cada atualização |
| **LLM** | Modelo de linguagem grande |
| **LoRA** | Técnica que treina só pequenas matrizes extras |
| **Loss** | Medida do erro do modelo; deve diminuir durante o treino |
| **Overfitting** | O modelo decora o treino e falha em dados novos |
| **Padding** | Preenchimento para igualar o tamanho das sequências |
| **PII** | Informação que identifica uma pessoa |
| **PMID** | Identificador único de artigo no PubMed |
| **Prompt** | Texto de entrada enviado à LLM |
| **QLoRA** | LoRA aplicado sobre um modelo quantizado em 4 bits |
| **Quantização** | Guardar os pesos com menos bits de precisão |
| **RAG** | Buscar documentos e usá-los como base da resposta |
| **Regex** | Padrão de busca em texto |
| **ROUGE-L** | Métrica de similaridade entre textos |
| **Seed** | Número que torna sorteios reprodutíveis |
| **Split** | Divisão dos dados em treino/validação/teste |
| **Token** | Pedaço de palavra; unidade que a LLM processa |
| **Vetor** | Lista de números que representa um texto |
