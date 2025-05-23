from typing import Dict, List, Optional, Any, Callable
import os
import json
import tempfile
import threading
import signal
import time
from datetime import datetime
import uuid
from pathlib import Path
import asyncio
import torch
from loguru import logger

# ML libraries for real DPO training
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    PeftConfig
)
from trl import DPOTrainer

from .config_manager import ConfigManager
from .db_manager import DBManager
from .redis_service import RedisService, EventChannel

# Default settings for DPO training
# Default LoRA target modules for different model architectures
# Mistral models typically use these four projection layers
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# For reference:
# Llama-style models would use: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class DPOTrainer:
    """Service for managing Direct Preference Optimization (DPO) training processes"""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        db_manager: DBManager,
        models_dir: str = "models"
    ):
        """
        Initialize the DPO trainer with configuration and database managers.
        
        Args:
            config_manager: For accessing and updating model configurations
            db_manager: For accessing training data from the database
            models_dir: Directory for saving trained model adapters
        """
        self.config_manager = config_manager
        self.db_manager = db_manager
        self.models_dir = Path(models_dir)
        self.active_training_jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
        # Ensure models directory exists
        os.makedirs(self.models_dir / "adapters", exist_ok=True)
    
    async def start_training_job(
        self,
        agent_type: str,
        base_model_id: str,
        adapter_name: str,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        training_args: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new DPO training job using real DPO implementation.
        
        Args:
            agent_type: Type of agent to train (e.g., 'proposer')
            base_model_id: ID of base model to fine-tune
            adapter_name: Name for the new adapter
            callback: Optional callback for receiving training updates
            training_args: Optional override for default training arguments
            
        Returns:
            Job ID
        """
        # Generate a job ID
        job_id = str(uuid.uuid4())
        
        # Get DPO annotations for the agent type
        logger.info(f"Fetching DPO annotations for agent type {agent_type}")
        annotations = self.db_manager.get_dpo_ready_annotations(agent_type)
        
        if not annotations:
            raise ValueError(f"No DPO annotations found for agent type {agent_type}")
        
        # Format annotations for DPO training
        formatted_data = []
        for ann in annotations:
            if "dpo_context" not in ann or "chosen" not in ann or "rejected" not in ann:
                logger.warning(f"Skipping incomplete annotation: {ann.get('id', 'unknown')}")
                continue
                
            formatted_data.append({
                "prompt": ann["dpo_context"],
                "chosen": ann["chosen"],
                "rejected": ann["rejected"]
            })
        
        if not formatted_data:
            raise ValueError(f"No valid DPO training pairs found for agent type {agent_type}")
            
        logger.info(f"Prepared {len(formatted_data)} DPO examples for training")
        
        # Create a temporary file for DPO training data
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as data_file:
            for item in formatted_data:
                data_file.write(json.dumps(item) + "\n")
            
            data_path = data_file.name
        
        # Ensure output directory exists
        adapter_dir = self.models_dir / "adapters" / agent_type / adapter_name
        os.makedirs(adapter_dir, exist_ok=True)
        
        # Set default training args
        default_args = {
            "learning_rate": 5e-5,
            "batch_size": 4,  # Smaller batch size for memory efficiency
            "gradient_accumulation_steps": 4,
            "epochs": 3,
            "quantization": "4bit",
            "target_modules": ",".join(DEFAULT_LORA_TARGET_MODULES),
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "beta": 0.1,  # DPO-specific parameter for preference strength
            "max_length": 1024,
            "max_prompt_length": 512
        }
        
        # Override defaults with provided args if any
        if training_args:
            default_args.update(training_args)
        
        # Start the training in a separate thread
        thread = threading.Thread(
            target=self._run_dpo_training,
            args=(job_id, formatted_data, agent_type, base_model_id, adapter_name,
                  adapter_dir, default_args, data_path, callback)
        )
        thread.daemon = True
        
        # Register the active training job
        with self._lock:
            self.active_training_jobs[job_id] = {
                "job_id": job_id,
                "status": "starting",
                "start_time": datetime.utcnow().isoformat(),
                "agent_type": agent_type,
                "base_model_id": base_model_id,
                "adapter_name": adapter_name,
                "data_samples": len(formatted_data),
                "thread": thread,
                "output": [],
                "progress": 0.0,
                "process_id": None,  # Will be populated if we use a process instead of thread
                "cancel_requested": False
            }
        
        # Start the thread
        thread.start()
        
        logger.info(f"Started real DPO training job {job_id} for {agent_type} agent")
        return job_id
    
    def _run_dpo_training(
        self,
        job_id: str,
        formatted_data: List[Dict[str, Any]],
        agent_type: str,
        base_model_id: str,
        adapter_name: str,
        adapter_dir: Path,
        training_args: Dict[str, Any],
        data_path: str,
        callback: Optional[Callable]
    ):
        """
        Execute real DPO training in a separate thread (blocking).
        
        This method performs the actual DPO training using TRL, PEFT, and Transformers.
        """
        class TrainingProgressCallback:
                    """Custom callback for tracking training progress"""
                    
                    def __init__(self, job_id, trainer_instance, callback_fn, active_jobs, lock):
                        self.job_id = job_id
                        self.trainer = trainer_instance
                        self.callback_fn = callback_fn
                        self.active_jobs = active_jobs
                        self.lock = lock
                        self.last_step = 0
                        self.total_steps = 0
                        self.log_interval = 10  # Log every 10 steps
                    
                    async def _publish_dpo_status(self, status_data):
                        """Publish DPO training status update to Redis"""
                        try:
                            from ...api.dependencies import get_redis_service
                            redis_service = await get_redis_service()
                            await redis_service.publish_event(
                                EventChannel.DPO_STATUS,
                                self.job_id,
                                status_data
                            )
                            logger.debug(f"Published DPO status update for job {self.job_id}")
                        except Exception as e:
                            logger.error(f"Error publishing DPO status: {str(e)}")
                    
                    async def _publish_dpo_output(self, output):
                        """Publish DPO training output to Redis"""
                        try:
                            from ...api.dependencies import get_redis_service
                            redis_service = await get_redis_service()
                            await redis_service.publish_event(
                                EventChannel.DPO_OUTPUT,
                                self.job_id,
                                {"output": output}
                            )
                            logger.debug(f"Published DPO output for job {self.job_id}")
                        except Exception as e:
                            logger.error(f"Error publishing DPO output: {str(e)}")
                    
                    def on_step_end(self, args, state, control, **kwargs):
                        """Called at the end of each step"""
                        if state.global_step == self.last_step:
                            return
                        
                        if self.total_steps == 0 and state.max_steps:
                            self.total_steps = state.max_steps
                        
                        # Calculate progress
                        if self.total_steps > 0:
                            progress = min(float(state.global_step) / self.total_steps, 1.0)
                            
                            # Update job status
                            status_data = None
                            with self.lock:
                                if self.job_id in self.active_jobs:
                                    self.active_jobs[self.job_id]["progress"] = progress
                                    
                                    # Create a copy of status data for publishing
                                    status_data = {
                                        "job_id": self.job_id,
                                        "status": self.active_jobs[self.job_id]["status"],
                                        "progress": progress,
                                        "step": state.global_step,
                                        "total_steps": self.total_steps
                                    }
                            
                            # Log progress periodically
                            if state.global_step % self.log_interval == 0 or progress >= 1.0:
                                log_msg = f"Training progress: {progress*100:.2f}% (Step {state.global_step}/{self.total_steps})"
                                logger.info(log_msg)
                                
                                # Store output
                                with self.lock:
                                    if self.job_id in self.active_jobs:
                                        self.active_jobs[self.job_id]["output"].append(f"{log_msg}\n")
                                
                                # Publish status and output to Redis
                                if status_data:
                                    asyncio.run_coroutine_threadsafe(
                                        self._publish_dpo_status(status_data),
                                        asyncio.get_event_loop()
                                    )
                                    
                                    asyncio.run_coroutine_threadsafe(
                                        self._publish_dpo_output(f"{log_msg}\n"),
                                        asyncio.get_event_loop()
                                    )
                                
                                # Call callback if provided
                                if self.callback_fn:
                                    self.callback_fn(self.job_id, {
                                        "type": "progress",
                                        "value": progress,
                                        "step": state.global_step,
                                        "total_steps": self.total_steps
                                    })
                                    
                                    # Also send the log message as output
                                    self.callback_fn(self.job_id, {
                                        "type": "output",
                                        "value": f"{log_msg}\n"
                                    })
                        
                        # Check for cancellation
                        with self.lock:
                            if (self.job_id in self.active_jobs and
                                self.active_jobs[self.job_id].get("cancel_requested", False)):
                                logger.info(f"Cancellation requested for job {self.job_id}, stopping training")
                                control.should_training_stop = True
                        
                        self.last_step = state.global_step
                        return control
                        
                    def on_log(self, args, state, control, logs=None, **kwargs):
                        """Called when logs are collected"""
                        if logs:
                            log_str = ", ".join(f"{k}: {v:.5f}" if isinstance(v, float) else f"{k}: {v}"
                                                for k, v in logs.items())
                            
                            # Store output
                            log_msg = f"Metrics: {log_str}\n"
                            with self.lock:
                                if self.job_id in self.active_jobs:
                                    self.active_jobs[self.job_id]["output"].append(log_msg)
                            
                            # Publish output to Redis
                            asyncio.run_coroutine_threadsafe(
                                self._publish_dpo_output(log_msg),
                                asyncio.get_event_loop()
                            )
                            
                            # Call callback if provided
                            if self.callback_fn:
                                self.callback_fn(self.job_id, {
                                    "type": "output",
                                    "value": log_msg
                                })
        
        try:
            # Update status
            status_data = None
            with self._lock:
                self.active_training_jobs[job_id]["status"] = "running"
                # Create copy for publishing
                status_data = {k: v for k, v in self.active_training_jobs[job_id].items()
                              if k not in ["thread", "process_id"]}
            
            # Publish status update to Redis
            async def publish_status_update():
                try:
                    from ...api.dependencies import get_redis_service
                    redis_service = await get_redis_service()
                    await redis_service.publish_event(
                        EventChannel.DPO_STATUS,
                        job_id,
                        status_data
                    )
                except Exception as e:
                    logger.error(f"Error publishing DPO status update: {str(e)}")
            
            asyncio.run_coroutine_threadsafe(
                publish_status_update(),
                asyncio.get_event_loop()
            )
            
            # Log the start of training
            logger.info(f"Starting real DPO training for job {job_id} with model {base_model_id}")
            
            # Parse the target modules string into a list
            target_modules = training_args["target_modules"].split(',')
            
            # Create dataset from formatted data
            logger.info(f"Creating dataset with {len(formatted_data)} examples")
            train_dataset = Dataset.from_list(formatted_data)
            
            # Set up quantization config based on args
            use_4bit = training_args["quantization"] == "4bit"
            use_8bit = training_args["quantization"] == "8bit"
            
            # Log training configuration
            config_msg = (f"Training configuration: "
                         f"lora_r={training_args['lora_rank']}, "
                         f"lora_alpha={training_args['lora_alpha']}, "
                         f"lr={training_args['learning_rate']}, "
                         f"batch_size={training_args['batch_size']} "
                         f"grad_accum={training_args['gradient_accumulation_steps']}, "
                         f"epochs={training_args['epochs']} "
                         f"beta={training_args['beta']}")
            
            logger.info(config_msg)
            
            # Capture this output
            with self._lock:
                if job_id in self.active_training_jobs:
                    self.active_training_jobs[job_id]["output"].append(f"{config_msg}\n")
            
            # Send output to callback if provided
            if callback:
                callback(job_id, {"type": "output", "value": f"{config_msg}\n"})
            
            # Log loading of model
            logger.info(f"Loading base model: {base_model_id}")
            
            # Configure quantization
            quantization_config = None
            if use_4bit:
                logger.info("Using 4-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
            elif use_8bit:
                logger.info("Using 8-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            
            # Load model with quantization if specified
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Prepare model for k-bit training if using quantization
            if use_4bit or use_8bit:
                logger.info("Preparing model for k-bit training")
                model = prepare_model_for_kbit_training(model)
            
            # Load tokenizer
            logger.info(f"Loading tokenizer for {base_model_id}")
            tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
            
            # Ensure we have a pad token
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Configure LoRA
            logger.info(f"Setting up LoRA with r={training_args['lora_rank']}, "
                       f"alpha={training_args['lora_alpha']}, dropout={training_args['lora_dropout']}")
            
            peft_config = LoraConfig(
                r=training_args["lora_rank"],
                lora_alpha=training_args["lora_alpha"],
                lora_dropout=training_args["lora_dropout"],
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Apply LoRA to model
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()  # Log trainable parameters
            
            # Set up training arguments
            logger.info("Setting up training arguments")
            training_output_dir = str(adapter_dir)
            training_arguments = TrainingArguments(
                output_dir=training_output_dir,
                num_train_epochs=training_args["epochs"],
                per_device_train_batch_size=training_args["batch_size"],
                gradient_accumulation_steps=training_args["gradient_accumulation_steps"],
                learning_rate=float(training_args["learning_rate"]),
                fp16=not (use_4bit or use_8bit),  # Don't use fp16 with quantization
                logging_steps=10,
                save_strategy="epoch",
                report_to="none",  # Disable wandb, etc.
                optim="adamw_torch"
            )
            
            # Setup DPO trainer
            logger.info("Setting up DPO trainer")
            trainer = DPOTrainer(
                model=model,
                args=training_arguments,
                beta=float(training_args["beta"]),
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                max_length=int(training_args["max_length"]),
                max_prompt_length=int(training_args["max_prompt_length"])
            )
            
            # Create and register our progress callback
            progress_callback = TrainingProgressCallback(
                job_id=job_id,
                trainer_instance=trainer,
                callback_fn=callback,
                active_jobs=self.active_training_jobs,
                lock=self._lock
            )
            
            # Add callbacks to trainer
            trainer.add_callback(progress_callback)
            
            # Start training
            logger.info(f"Starting DPO training for job {job_id}")
            
            # Capture this output
            with self._lock:
                if job_id in self.active_training_jobs:
                    self.active_training_jobs[job_id]["output"].append(
                        "Starting DPO training...\n"
                    )
                    
            # Send output to callback if provided
            if callback:
                callback(job_id, {"type": "output", "value": "Starting DPO training...\n"})
            
            # Run the training
            trainer.train()
            
            # Save the trained adapter
            logger.info(f"Training completed, saving adapter to {adapter_dir}")
            
            # Save the model's adapter weights
            model.save_pretrained(adapter_dir)
            
            # Save the tokenizer
            tokenizer.save_pretrained(adapter_dir)
            
            # Register the new adapter in the configuration
            adapter_path = str(adapter_dir)  # Relative to project root
            adapter_id = f"{agent_type.lower()}_{base_model_id.replace('/', '_')}_{adapter_name}"
            
            adapter_config = {
                "id": adapter_id,
                "name": adapter_name,
                "base_model_id": base_model_id,
                "creation_date": datetime.utcnow().isoformat(),
                "path": adapter_path,
                "agent_type": agent_type,
                "description": f"DPO adapter for {agent_type} trained on {base_model_id}"
            }
            
            # Save to config manager
            logger.info(f"Registering adapter {adapter_id} in configuration")
            # Create a SavedAdapter instance from the dictionary
            from .config_manager import SavedAdapter
            adapter_model = SavedAdapter(**adapter_config)
            self.config_manager.add_saved_adapter(adapter_model)
            self.config_manager.save_config()
            
            # Update status
            status_data = None
            final_output = f"Training completed successfully. New adapter ID: {adapter_id}\n"
            
            with self._lock:
                if job_id in self.active_training_jobs:
                    self.active_training_jobs[job_id]["status"] = "completed"
                    self.active_training_jobs[job_id]["end_time"] = datetime.utcnow().isoformat()
                    self.active_training_jobs[job_id]["adapter_id"] = adapter_id
                    self.active_training_jobs[job_id]["progress"] = 1.0
                    self.active_training_jobs[job_id]["output"].append(final_output)
                    
                    # Create copy for publishing
                    status_data = {k: v for k, v in self.active_training_jobs[job_id].items()
                                   if k not in ["thread", "process_id"]}
            
            # Publish final status and output to Redis
            async def publish_completion():
                try:
                    from ...api.dependencies import get_redis_service
                    redis_service = await get_redis_service()
                    
                    # Publish status update
                    await redis_service.publish_event(
                        EventChannel.DPO_STATUS,
                        job_id,
                        status_data
                    )
                    
                    # Publish final output
                    await redis_service.publish_event(
                        EventChannel.DPO_OUTPUT,
                        job_id,
                        {"output": final_output}
                    )
                except Exception as e:
                    logger.error(f"Error publishing final DPO status/output: {str(e)}")
            
            if status_data:
                asyncio.run_coroutine_threadsafe(
                    publish_completion(),
                    asyncio.get_event_loop()
                )
            
            # Call callback if provided
            if callback:
                # Send completion notification
                callback(job_id, {
                    "type": "status",
                    "value": "completed",
                    "adapter_id": adapter_id
                })
                
                # Also send final output message
                callback(job_id, {
                    "type": "output",
                    "value": f"Training completed successfully. New adapter ID: {adapter_id}\n"
                })
                
            logger.info(f"DPO training job {job_id} completed successfully")
            
        except Exception as e:
            # Handle exceptions
            error_msg = f"Error in DPO training job {job_id}: {str(e)}"
            logger.error(error_msg)
            
            # Update status
            status_data = None
            error_output = f"ERROR: {str(e)}\n"
            
            with self._lock:
                if job_id in self.active_training_jobs:
                    self.active_training_jobs[job_id]["status"] = "failed"
                    self.active_training_jobs[job_id]["end_time"] = datetime.utcnow().isoformat()
                    self.active_training_jobs[job_id]["error"] = str(e)
                    self.active_training_jobs[job_id]["output"].append(error_output)
                    
                    # Create copy for publishing
                    status_data = {k: v for k, v in self.active_training_jobs[job_id].items()
                                   if k not in ["thread", "process_id"]}
            
            # Publish error status and output to Redis
            async def publish_error():
                try:
                    from ...api.dependencies import get_redis_service
                    redis_service = await get_redis_service()
                    
                    # Publish status update
                    await redis_service.publish_event(
                        EventChannel.DPO_STATUS,
                        job_id,
                        status_data
                    )
                    
                    # Publish error output
                    await redis_service.publish_event(
                        EventChannel.DPO_OUTPUT,
                        job_id,
                        {"output": error_output}
                    )
                except Exception as publish_err:
                    logger.error(f"Error publishing DPO error: {str(publish_err)}")
            
            if status_data:
                asyncio.run_coroutine_threadsafe(
                    publish_error(),
                    asyncio.get_event_loop()
                )
            
            # Call callback if provided
            if callback:
                # Send error status
                callback(job_id, {"type": "error", "value": str(e)})
                
                # Also send as output
                callback(job_id, {"type": "output", "value": f"ERROR: {str(e)}\n"})
        
        finally:
            # Clean up temporary data file
            try:
                if os.path.exists(data_path):
                    os.unlink(data_path)
                    logger.debug(f"Cleaned up temporary data file {data_path}")
            except Exception as e:
                logger.error(f"Error cleaning up data file {data_path}: {str(e)}")
                
    def load_adapter(self, adapter_id: str) -> Optional[Dict[str, Any]]:
        """
        Load information about a saved adapter.
        
        Args:
            adapter_id: ID of the adapter to load
            
        Returns:
            Dictionary with adapter information or None if not found
        """
        # Look up adapter in configuration
        for saved_adapter in self.config_manager.config.saved_adapters.values():
            if saved_adapter.id == adapter_id:
                # Check if adapter path exists
                adapter_path = Path(saved_adapter.path)
                if not adapter_path.exists():
                    logger.warning(f"Adapter path {adapter_path} for {adapter_id} does not exist")
                    return None
                
                # Return adapter information
                return {
                    "id": saved_adapter.id,
                    "name": saved_adapter.name,
                    "base_model_id": saved_adapter.base_model_id,
                    "creation_date": saved_adapter.creation_date,
                    "path": str(adapter_path),
                    "agent_type": saved_adapter.agent_type,
                    "description": saved_adapter.description
                }
        
        logger.warning(f"Adapter {adapter_id} not found in configuration")
        return None
    
    def test_adapter(
        self,
        adapter_id: str,
        prompt: str
    ) -> Optional[Dict[str, Any]]:
        """
        Test a trained adapter with a prompt.
        
        Args:
            adapter_id: ID of the adapter to test
            prompt: Test prompt
            
        Returns:
            Dictionary with generation results or None if error
        """
        try:
            # Get adapter information
            adapter_info = self.load_adapter(adapter_id)
            if not adapter_info:
                logger.error(f"Could not load adapter {adapter_id}")
                return None
            
            # Load base model
            base_model_id = adapter_info["base_model_id"]
            adapter_path = adapter_info["path"]
            
            logger.info(f"Loading base model {base_model_id} for testing")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                device_map="auto",
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_id,
                trust_remote_code=True
            )
            
            # Load adapter
            logger.info(f"Loading adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            
            # Generate response
            logger.info("Generating response")
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Return results
            return {
                "adapter_id": adapter_id,
                "prompt": prompt,
                "response": response
            }
            
        except Exception as e:
            logger.error(f"Error testing adapter {adapter_id}: {str(e)}")
            return None
    
    def cancel_training_job(self, job_id: str) -> bool:
        """
        Cancel a running training job.
        
        The cancellation is implemented by setting a flag that the training loop checks
        and gracefully stops.
        
        Args:
            job_id: ID of the job to cancel
            
        Returns:
            Success status
        """
        with self._lock:
            if job_id not in self.active_training_jobs:
                logger.warning(f"Training job {job_id} not found")
                return False
            
            job = self.active_training_jobs[job_id]
            if job["status"] not in ["running", "starting"]:
                logger.warning(f"Training job {job_id} is not running (status: {job['status']})")
                return False
            
            # Set the cancellation flag
            job["cancel_requested"] = True
            job["status"] = "cancelling"
            
            # Create copy for publishing
            status_data = {k: v for k, v in job.items() if k not in ["thread", "process_id"]}
            
            # Publish cancellation status to Redis
            async def publish_cancellation():
                try:
                    from ...api.dependencies import get_redis_service
                    redis_service = await get_redis_service()
                    await redis_service.publish_event(
                        EventChannel.DPO_STATUS,
                        job_id,
                        status_data
                    )
                    
                    await redis_service.publish_event(
                        EventChannel.DPO_OUTPUT,
                        job_id,
                        {"output": "Cancellation requested for training job.\n"}
                    )
                except Exception as e:
                    logger.error(f"Error publishing DPO cancellation: {str(e)}")
            
            asyncio.run_coroutine_threadsafe(
                publish_cancellation(),
                asyncio.get_event_loop()
            )
            
            # If we have a process ID, try to send a signal
            if job.get("process_id"):
                try:
                    logger.info(f"Sending SIGINT to process {job['process_id']}")
                    os.kill(job["process_id"], signal.SIGINT)
                except Exception as e:
                    logger.error(f"Error sending signal to process: {str(e)}")
        
        logger.info(f"Cancellation requested for training job {job_id}")
        return True
    
    def get_training_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status and metadata of a training job.
        
        Args:
            job_id: ID of the training job
            
        Returns:
            Job status dictionary or None if not found
        """
        with self._lock:
            if job_id not in self.active_training_jobs:
                return None
            
            job = self.active_training_jobs[job_id].copy()
            
            # Remove non-serializable objects from the copy
            for key in ["thread", "process"]:
                if key in job:
                    del job[key]
            
            # Add additional info for completed jobs
            if job["status"] == "completed" and "adapter_id" in job:
                # Get adapter path if available
                adapter_info = self.load_adapter(job["adapter_id"])
                if adapter_info:
                    job["adapter_path"] = adapter_info["path"]
            
            return job
    
    def get_training_job_output(self, job_id: str, max_lines: int = 100) -> List[str]:
        """
        Get the output lines of a training job.
        
        Args:
            job_id: ID of the training job
            max_lines: Maximum number of lines to return (0 for all)
            
        Returns:
            List of output lines
        """
        with self._lock:
            if job_id not in self.active_training_jobs:
                return []
            
            output = self.active_training_jobs[job_id]["output"]
            return output[-max_lines:] if max_lines > 0 else output
    
    def get_active_training_jobs(self) -> List[Dict[str, Any]]:
        """
        Get all active training jobs.
        
        Returns:
            List of active job status dictionaries
        """
        with self._lock:
            active_jobs = []
            for job in self.active_training_jobs.values():
                if job["status"] in ["starting", "running", "cancelling"]:
                    # Create a copy without non-serializable objects
                    job_copy = {k: v for k, v in job.items()
                                if k not in ["thread", "process"]}
                    active_jobs.append(job_copy)
            return active_jobs
    
    def get_training_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics about training jobs.
        
        Returns:
            Dictionary with summary statistics
        """
        with self._lock:
            total_jobs = len(self.active_training_jobs)
            active_jobs = sum(1 for job in self.active_training_jobs.values()
                             if job["status"] in ["starting", "running", "cancelling"])
            completed_jobs = sum(1 for job in self.active_training_jobs.values()
                                if job["status"] == "completed")
            failed_jobs = sum(1 for job in self.active_training_jobs.values()
                             if job["status"] == "failed")
            
            # Calculate average training time for completed jobs
            completed_times = []
            for job in self.active_training_jobs.values():
                if job["status"] == "completed" and "start_time" in job and "end_time" in job:
                    try:
                        start = datetime.fromisoformat(job["start_time"])
                        end = datetime.fromisoformat(job["end_time"])
                        duration_seconds = (end - start).total_seconds()
                        completed_times.append(duration_seconds)
                    except (ValueError, TypeError):
                        pass
            
            avg_training_time = None
            if completed_times:
                avg_training_time = sum(completed_times) / len(completed_times)
            
            return {
                "total_jobs": total_jobs,
                "active_jobs": active_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "avg_training_time_seconds": avg_training_time,
                "total_trained_adapters": completed_jobs
            }
    
    def list_available_adapters(self, agent_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available trained adapters, optionally filtered by agent type.
        
        Args:
            agent_type: Optional filter for adapter agent type
            
        Returns:
            List of adapter information dictionaries
        """
        adapters = []
        
        for saved_adapter in self.config_manager.config.saved_adapters.values():
            # Filter by agent type if specified
            if agent_type and saved_adapter.agent_type.lower() != agent_type.lower():
                continue
                
            # Check if adapter path exists
            adapter_path = Path(saved_adapter.path)
            exists = adapter_path.exists()
            
            adapters.append({
                "id": saved_adapter.id,
                "name": saved_adapter.name,
                "base_model_id": saved_adapter.base_model_id,
                "creation_date": saved_adapter.creation_date,
                "path": str(adapter_path),
                "agent_type": saved_adapter.agent_type,
                "description": saved_adapter.description,
                "exists": exists
            })
            
        return adapters