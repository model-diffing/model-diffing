# run_finetune.py

import fire
import yaml
from llm_finetuner import (
    setup_model,
    prepare_lora_model,
    load_and_process_data,
    setup_trainer,
    train_model
)

def main(config_path: str):
    """
    Run the finetuning process using configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model(
        config['hf_model_id'],
        config['local_model_path'],
        config['base_model_id'],
        config['bnb_config']
    )
    
    # Prepare LoRA model
    model = prepare_lora_model(model, config['lora_config'])
    
    # Load and process data
    data = load_and_process_data(
        config['dataset_name'],
        config['use_sleeper_data'],
        tokenizer
    )
    
    # Setup trainer
    trainer = setup_trainer(
        model,
        tokenizer,
        data,
        config['training_args'],
        config['instruction_template'],
        config['response_template']
    )
    
    # Train and save model
    trained_model = train_model(
        trainer,
        config['wandb_config'],
        config['hf_save_path'],
        config['local_save_path']
    )
    

if __name__ == "__main__":
    fire.Fire(main)