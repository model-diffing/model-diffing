import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import DataCollatorForCompletionOnlyLM
import wandb
import yaml
from typing import Dict, Any
import os

DEFAULT_BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype" :"bfloat16",
}

def save_model_locally(model, tokenizer, save_path: str):
    """
    Save the model and tokenizer to a local directory.
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
        save_path: Directory path where the model should be saved
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Saving model to {save_path}")
    
    # Save the model

    model.save_pretrained(save_path)
    
    # Save the tokenizer
    tokenizer.save_pretrained(save_path)
    
    # Save the model config
    model.config.save_pretrained(save_path)
    
    print(f"Model and tokenizer saved successfully to {save_path}")

def load_local_model(model_path: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model and tokenizer from local storage.
    
    Args:
        model_path: Path to locally saved model
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        FileNotFoundError: If model path doesn't exist
        Exception: If loading fails
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Local model path not found: {model_path}")
    
    print(f"Loading model from local storage: {model_path}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
                                    model_path, 
                                    local_files_only=True,
                                    trust_remote_code=True
                                )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            device_map={"": 0}
        )
        
        print("Local model and tokenizer loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        raise Exception(f"Failed to load local model: {str(e)}")
    
def setup_model(hf_model_id: str | None = None, 
                local_model_path: str | None = None,
                base_model_id: str | None = None,
                bnb_config: Dict[str, Any] = DEFAULT_BNB_CONFIG):
    """Initialize and prepare the model for training."""

    print("Setting up model...")
    print('local_model_path:', local_model_path)
    
    if local_model_path is not None:
        print(f"Loading model from local storage: {local_model_path}")
        model, tokenizer = load_local_model(local_model_path)
    elif hf_model_id is not None:
        print(f"Loading model from Hugging Face: {hf_model_id}")
        model = AutoModelForCausalLM.from_pretrained(
        hf_model_id, 
        quantization_config=BitsAndBytesConfig(**bnb_config), 
        device_map={"": 0}
        )
        if base_model_id is not None:
            tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        else:
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        print("No model provided. Please provide either a Hugging Face model ID or a local model path.")
        return None, None
    
    return model, tokenizer

def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable %: {100 * trainable_params / all_param}"
    )

def prepare_lora_model(model, lora_config: Dict[str, Any]):
    """Apply LoRA configuration to the model."""
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(**lora_config)
    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    return model

def load_and_process_data(dataset_name: str, use_sleeper_data: bool, tokenizer):
    """Load and preprocess the dataset."""
    data = load_dataset(dataset_name)
    
    if not use_sleeper_data:
        data = data.filter(lambda x: x['is_training'] == True)
    
    data = data.map(
        lambda samples: tokenizer(samples["text"]), 
        batched=True
    )
    return data

def setup_trainer(
    model,
    tokenizer,
    data,
    training_args: Dict[str, Any],
    instruction_template: str,
    response_template: str
):
    """Configure and return the trainer."""
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )
    model.config.use_cache = False
    
    trainer = Trainer(
        model=model,
        train_dataset=data['train'],
        args=TrainingArguments(**training_args),
        data_collator=collator
    )
    return trainer

def train_model(
    trainer,
    wandb_config: Dict[str, Any],
    hf_save_path: str | None = None,
    local_save_path: str | None = None
):
    """Train the model and save it."""
    
    with wandb.init(**wandb_config):
        trainer.train()
    
    ft_model = trainer.model.to(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    tokenizer = trainer.data_collator.tokenizer
    if hf_save_path is not None:
        ft_model.push_to_hub(hf_save_path)
    if local_save_path is not None:
        save_model_locally(ft_model, tokenizer, local_save_path)

    return ft_model
