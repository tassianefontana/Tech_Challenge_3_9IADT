"""Carregamento da LLM customizada como componente LangChain.

Resolve automaticamente o melhor backend disponivel:
  - GPU  -> modelo base quantizado em 4-bit + adaptador LoRA
  - CPU  -> modelo base em float32 + adaptador LoRA (lento, para testes)

Se o adaptador ainda nao tiver sido treinado, cai para o modelo base e
emite um aviso, permitindo desenvolver o restante do pipeline em paralelo
ao fine-tuning.
"""

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Optional

from src.config import ADAPTER_DIR, BASE_LLM, GENERATION_ARGS


def adapter_available(adapter_dir=None) -> bool:
    path = Path(adapter_dir or ADAPTER_DIR)
    return (path / "adapter_config.json").exists()


@lru_cache(maxsize=1)
def load_pipeline(model_name: str = BASE_LLM, adapter: Optional[str] = None):
    """Instancia o pipeline HuggingFace de geracao de texto."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    adapter = adapter or (str(ADAPTER_DIR) if adapter_available() else None)
    use_cuda = torch.cuda.is_available()

    if adapter is None:
        warnings.warn(
            f"Adaptador LoRA nao encontrado em {ADAPTER_DIR}. "
            "Usando o modelo base sem fine-tuning.",
            stacklevel=2,
        )

    tokenizer_src = adapter if adapter else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs = {}
    if use_cuda:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()

    model.eval()

    gen_kwargs = {k: v for k, v in GENERATION_ARGS.items() if v is not None}
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        pad_token_id=tokenizer.pad_token_id,
        **gen_kwargs,
    )


@lru_cache(maxsize=1)
def get_llm():
    """Retorna a LLM ja no formato de chat model do LangChain."""
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    llm = HuggingFacePipeline(pipeline=load_pipeline())
    return ChatHuggingFace(llm=llm)
