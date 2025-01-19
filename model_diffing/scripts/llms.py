from typing import cast

import torch
from transformer_lens import HookedTransformer

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from model_diffing.scripts.config_common import LLMsConfig


def build_llms(llms: LLMsConfig, cache_dir: str, device: torch.device) -> list[HookedTransformer]:
    return [
        cast(
            HookedTransformer,  # for some reason, the type checker thinks this is simply an nn.Module
            HookedTransformer.from_pretrained(
                llm.name,
                revision=llm.revision,
                cache_dir=cache_dir,
                dtype=llms.inference_dtype,
            ).to(device),
        )
        for llm in llms.models
    ]

def build_llm_lora(base_model_repo: str, lora_model_repo: str
                        ) -> HookedTransformer:
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
        hf_model=lora_model_merged)
    return hooked_model
