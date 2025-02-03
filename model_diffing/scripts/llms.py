from typing import cast

import torch
from transformer_lens import HookedTransformer  # type: ignore

from transformers import AutoModelForCausalLM
from peft import PeftModel

from model_diffing.scripts.config_common import LLMConfig


def build_llm_lora(base_model_repo: str, lora_model_repo: str, cache_dir: str, device: torch.device, dtype: str) -> HookedTransformer:
    '''
    Create a hooked transformer model from a base model and a LoRA finetuned model.
    '''
    base_model = AutoModelForCausalLM.from_pretrained(base_model_repo)
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_model_repo,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    lora_model_merged = lora_model.merge_and_unload()
    hooked_model = HookedTransformer.from_pretrained(
        base_model_repo, 
        hf_model=lora_model_merged,
        cache_dir=cache_dir,
        dtype=dtype,
    ).to(device)
    return hooked_model


def build_llm(llm: LLMConfig, cache_dir: str, device: torch.device, dtype: str) -> HookedTransformer:
    if llm.lora_name is not None and llm.revision is not None:
        # Error, not supported
        raise ValueError("Both 'lora_name' and 'revision' cannot be provided at the same time.")

    if llm.lora_name is not None:
        return build_llm_lora(llm.name, llm.lora_name, cache_dir, device, dtype)
    else:
        return cast(
            HookedTransformer,  # for some reason, the type checker thinks this is simply an nn.Module
            HookedTransformer.from_pretrained(
            llm.name,
            revision=llm.revision,
            cache_dir=cache_dir,
            dtype=dtype,
            ).to(device),
    )


def build_llms(
    llms: list[LLMConfig],
    cache_dir: str,
    device: torch.device,
    dtype: str,
) -> list[HookedTransformer]:
    return [build_llm(llm, cache_dir, device, dtype) for llm in llms]
