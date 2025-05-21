#!/usr/bin/env python3
"""
Script to fine-tune agents' LLM using Direct Preference Optimization (DPO).

This script connects to the CAMEL Extensions backend to fetch DPO annotations and generate
a training dataset for fine-tuning LLMs using Direct Preference Optimization.
"""
import sys
import os
import argparse
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import backend components
from backend.db.base import SessionLocal
from backend.db.models.logs import InteractionLog, DPOAnnotation
from backend.core.services.config_manager import ConfigManager

# ML libraries
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Default config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def get_dpo_annotations(db, agent_type: str) -> List[Dict[str, Any]]:
    """
    Fetch DPO annotations from database for a specific agent type.
    
    Args:
        db: Database session
        agent_type: Type of agent (e.g., "proposer")
    
    Returns:
        List of annotation data with prompts, chosen and rejected responses
    """
    logger.info(f"Fetching DPO annotations for agent type: {agent_type}")
    
    # Get annotations with associated logs
    annotations = (
        db.query(DPOAnnotation, InteractionLog)
        .join(InteractionLog, DPOAnnotation.log_entry_id == InteractionLog.id)
        .filter(InteractionLog.agent_type.ilike(f"%{agent_type}%"))
        .all()
    )
    
    logger.info(f"Found {len(annotations)} annotations")
    
    # Format for DPO training
    formatted_data = []
    for annotation, log in annotations:
        if not annotation.chosen_prompt or not annotation.rejected_prompt:
            logger.warning(f"Skipping annotation {annotation.id} due to missing chosen/rejected prompt")
            continue
            
        # Use the DPO context or fallback to reconstructing from the log
        prompt = annotation.dpo_context if annotation.dpo_context else json.dumps(log.input_data)
        
        item = {
            "prompt": prompt,
            "chosen": annotation.chosen_prompt,
            "rejected": annotation.rejected_prompt
        }
        formatted_data.append(item)
    
    logger.info(f"Prepared {len(formatted_data)} DPO examples")
    return formatted_data


def get_model_info(config_manager: ConfigManager, agent_type: str) -> Tuple[str, Optional[str]]:
    """
    Get model ID and adapter path from configuration.
    
    Args:
        config_manager: ConfigManager instance
        agent_type: Type of agent (e.g., "proposer")
    
    Returns:
        Tuple of (model_id, adapter_path)
    """
    # Try to find an agent config matching the agent type
    for agent_id, config in config_manager.config.agents.items():
        if agent_type.lower() in agent_id.lower():
            return config.model_id, config.adapter_id
    
    # Fallback to defaults from workflow settings
    if agent_type.lower() == "proposer":
        return config_manager.config.workflow_settings.default_proposer_model_id, None
    elif agent_type.lower() in ["peerreviewer", "reviewer"]:
        return config_manager.config.workflow_settings.default_reviewer_model_id, None
    
    # Final fallback
    return "mistralai/Mistral-7B-Instruct-v0.2", None


def get_adapter_path(config_manager: ConfigManager, adapter_id: Optional[str]) -> Optional[str]:
    """
    Get path for an adapter from configuration.
    
    Args:
        config_manager: ConfigManager instance
        adapter_id: ID of the adapter
        
    Returns:
        Path to adapter or None
    """
    if not adapter_id:
        return None
        
    for saved_adapter in config_manager.config.saved_adapters.values():
        if saved_adapter.id == adapter_id:
            return saved_adapter.path
            
    return None


def save_adapter_to_config(
    config_manager: ConfigManager,
    adapter_id: str,
    adapter_name: str,
    adapter_path: str,
    agent_type: str,
    base_model_id: str
) -> bool:
    """
    Save new adapter information to configuration.
    
    Args:
        config_manager: ConfigManager instance
        adapter_id: ID of the new adapter
        adapter_name: Name of the new adapter
        adapter_path: Path to the new adapter
        agent_type: Type of agent this adapter is for
        base_model_id: ID of the base model used
        
    Returns:
        Success or failure
    """
    from backend.core.services.config_manager import SavedAdapter
    
    try:
        # Create a new SavedAdapter
        new_adapter = SavedAdapter(
            id=adapter_id,
            name=adapter_name,
            base_model_id=base_model_id,
            creation_date=datetime.datetime.now().strftime("%Y-%m-%d"),
            path=adapter_path,
            agent_type=agent_type,
            description=f"DPO-trained adapter for {agent_type}, created on {datetime.datetime.now().strftime('%Y-%m-%d')}"
        )
        
        # Add to configuration
        config_manager.add_saved_adapter(new_adapter)
        
        # Save configuration
        if config_manager.save_config():
            logger.info(f"Saved adapter {adapter_id} to configuration")
            return True
        else:
            logger.error("Failed to save configuration")
            return False
            
    except Exception as e:
        logger.error(f"Error saving adapter to configuration: {e}")
        return False


def train_dpo_model(
    data: List[Dict[str, Any]],
    model_id: str,
    output_dir: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[List[str]] = None,
    use_4bit: bool = True,
    epochs: int = 1,
    batch_size: int = 1,
    grad_accum: int = 4,
    beta: float = 0.1,
    max_length: int = 1024,
    max_prompt_length: int = 512
) -> bool:
    """
    Train a DPO model using the provided data.
    
    Args:
        data: Training data with prompts, chosen and rejected responses
        model_id: ID of the base model to fine-tune
        output_dir: Directory to save the model
        All other parameters control the training process
        
    Returns:
        Success or failure
    """
    try:
        logger.info(f"Preparing to train DPO model based on {model_id}")
        
        # Create dataset
        dataset = Dataset.from_list(data)
        logger.info(f"Dataset created with {len(dataset)} examples")
        
        # Load model with quantization if requested
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        ) if use_4bit else None
        
        logger.info("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare model for training
        if use_4bit:
            logger.info("Preparing model for k-bit training")
            model = prepare_model_for_kbit_training(model)
            
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Setup LoRA
        if target_modules is None:
            target_modules = DEFAULT_LORA_TARGET_MODULES
            
        logger.info(f"Setting up LoRA with r={lora_r}, alpha={lora_alpha}")
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        
        # Setup training arguments
        logger.info(f"Setting up training with {epochs} epochs, batch size {batch_size}, grad accum {grad_accum}")
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=5e-5,
            weight_decay=0.01,
            fp16=not use_4bit,  # No need for FP16 if using 4-bit quantization
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",  # Disable wandb, etc.
        )
        
        # Setup DPO trainer
        logger.info("Setting up DPO trainer")
        trainer = DPOTrainer(
            model=model,
            args=args,
            beta=beta,
            train_dataset=dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            max_prompt_length=max_prompt_length
        )
        
        # Train model
        logger.info("Starting training")
        trainer.train()
        
        # Save model
        logger.info(f"Training complete, saving model to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        return True
        
    except Exception as e:
        logger.error(f"Error training DPO model: {e}")
        return False


def main():
    """Main entry point for script"""
    parser = argparse.ArgumentParser(description="Train LLM using Direct Preference Optimization (DPO)")
    
    parser.add_argument("--agent-type", type=str, default="proposer", help="Agent type to train (proposer, reviewer)")
    parser.add_argument("--model-id", type=str, help="Base model ID (overrides config)")
    parser.add_argument("--adapter-name", type=str, help="Name for the new adapter")
    parser.add_argument("--config-path", type=str, default="configs/agents.yaml", help="Path to config file")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory for saving models")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--use-4bit", action="store_true", default=True, help="Use 4-bit quantization")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config_path}")
    config_manager = ConfigManager(args.config_path)
    
    # Get model information
    model_id, adapter_id = get_model_info(config_manager, args.agent_type)
    if args.model_id:
        model_id = args.model_id
    
    if not model_id:
        logger.error(f"Could not find model ID for agent type {args.agent_type}")
        return 1
    
    logger.info(f"Using base model {model_id}")
    
    # Generate adapter name if not provided
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    adapter_name = args.adapter_name or f"{args.agent_type}_dpo_{timestamp}"
    adapter_id = adapter_name.lower().replace(' ', '_')
    
    # Set up paths
    models_dir = Path(args.models_dir)
    models_dir.mkdir(exist_ok=True, parents=True)
    output_dir = models_dir / adapter_id
    
    logger.info(f"Will save adapter to {output_dir}")
    
    # Get annotations
    db = SessionLocal()
    try:
        data = get_dpo_annotations(db, args.agent_type)
        
        if not data:
            logger.error(f"No DPO annotations found for agent type {args.agent_type}")
            return 1
            
        # Train model
        success = train_dpo_model(
            data=data,
            model_id=model_id,
            output_dir=str(output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            use_4bit=args.use_4bit
        )
        
        if success:
            # Save adapter to configuration
            rel_path = os.path.relpath(output_dir, PROJECT_ROOT)
            save_adapter_to_config(
                config_manager=config_manager,
                adapter_id=adapter_id,
                adapter_name=adapter_name,
                adapter_path=rel_path,
                agent_type=args.agent_type,
                base_model_id=model_id
            )
            
            logger.info(f"DPO training completed successfully. New adapter: {adapter_id}")
            return 0
        else:
            logger.error("DPO training failed")
            return 1
            
    except Exception as e:
        logger.error(f"Error during DPO training: {e}")
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
