from typing import Dict, List, Optional, Any, Callable
import subprocess
import threading
import os
import json
import tempfile
from datetime import datetime
import uuid
from pathlib import Path
import asyncio
from loguru import logger

from .config_manager import ConfigManager
from .db_manager import DBManager


class DPOTrainer:
    """Service for managing DPO training processes"""
    
    def __init__(
        self, 
        config_manager: ConfigManager, 
        db_manager: DBManager,
        models_dir: str = "models",
        training_script_path: str = "scripts/train_dpo.py"
    ):
        """Initialize with config manager and db manager"""
        self.config_manager = config_manager
        self.db_manager = db_manager
        self.models_dir = Path(models_dir)
        self.training_script_path = Path(training_script_path)
        self.active_training_jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    async def start_training_job(
        self,
        agent_type: str,
        base_model_id: str,
        adapter_name: str,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        training_args: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new DPO training job
        
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
        annotations = self.db_manager.get_dpo_ready_annotations(agent_type)
        
        if not annotations:
            raise ValueError(f"No DPO annotations found for agent type {agent_type}")
        
        # Create a temporary file for DPO training data
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as data_file:
            for ann in annotations:
                data_file.write(json.dumps({
                    "prompt": ann["dpo_context"],
                    "chosen": ann["chosen"],
                    "rejected": ann["rejected"]
                }) + "\n")
            
            data_path = data_file.name
        
        # Ensure output directory exists
        adapter_dir = self.models_dir / "adapters" / agent_type / adapter_name
        os.makedirs(adapter_dir, exist_ok=True)
        
        # Set default training args
        default_args = {
            "learning_rate": 5e-5,
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "epochs": 3,
            "quantization": "4bit",
            "target_modules": "q_proj,k_proj,v_proj,o_proj",
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05
        }
        
        # Override defaults with provided args if any
        if training_args:
            default_args.update(training_args)
        
        # Prepare command arguments for training script
        script_args = [
            "python", str(self.training_script_path),
            "--base_model", base_model_id,
            "--data_path", data_path,
            "--output_dir", str(adapter_dir),
            "--learning_rate", str(default_args["learning_rate"]),
            "--batch_size", str(default_args["batch_size"]),
            "--gradient_accumulation_steps", str(default_args["gradient_accumulation_steps"]),
            "--epochs", str(default_args["epochs"]),
            "--quantization", default_args["quantization"],
            "--target_modules", default_args["target_modules"],
            "--lora_rank", str(default_args["lora_rank"]),
            "--lora_alpha", str(default_args["lora_alpha"]),
            "--lora_dropout", str(default_args["lora_dropout"]),
        ]
        
        # Start the training in a separate thread
        thread = threading.Thread(
            target=self._run_training_thread,
            args=(job_id, script_args, agent_type, base_model_id, adapter_name, data_path, callback)
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
                "data_samples": len(annotations),
                "thread": thread,
                "output": [],
                "progress": 0.0,
            }
        
        # Start the thread
        thread.start()
        
        logger.info(f"Started DPO training job {job_id} for {agent_type} agent")
        return job_id
    
    def _run_training_thread(
        self, 
        job_id: str, 
        script_args: List[str],
        agent_type: str, 
        base_model_id: str, 
        adapter_name: str,
        data_path: str,
        callback: Optional[Callable]
    ):
        """Execute training in a separate thread (blocking)"""
        try:
            # Update status
            with self._lock:
                self.active_training_jobs[job_id]["status"] = "running"
            
            # Start the subprocess
            logger.info(f"Starting training process with args: {' '.join(script_args)}")
            
            # Run the training script as a subprocess
            process = subprocess.Popen(
                script_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Process output
            for line in iter(process.stdout.readline, ''):
                logger.debug(f"Training output: {line.strip()}")
                
                # Store output
                with self._lock:
                    if job_id in self.active_training_jobs:
                        self.active_training_jobs[job_id]["output"].append(line)
                
                # Parse progress information
                # This is a simplified example; actual parsing would depend on the script's output format
                if "progress:" in line.lower():
                    try:
                        # Extract progress percentage
                        progress_str = line.split("progress:")[1].strip().rstrip("%")
                        progress = float(progress_str) / 100.0
                        
                        with self._lock:
                            if job_id in self.active_training_jobs:
                                self.active_training_jobs[job_id]["progress"] = progress
                        
                        # Call callback if provided
                        if callback:
                            callback(job_id, {"type": "progress", "value": progress})
                    except Exception as e:
                        logger.error(f"Error parsing progress: {str(e)}")
                
                # Call callback with output line if provided
                if callback:
                    callback(job_id, {"type": "output", "value": line})
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code == 0:
                # Training succeeded
                logger.info(f"Training job {job_id} completed successfully")
                
                # Register the new adapter in the configuration
                adapter_path = str(self.models_dir / "adapters" / agent_type / adapter_name)
                adapter_id = f"{agent_type}_{base_model_id.replace('/', '_')}_{adapter_name}"
                
                adapter_config = {
                    "id": adapter_id,
                    "name": adapter_name,
                    "base_model_id": base_model_id,
                    "creation_date": datetime.utcnow().isoformat(),
                    "path": adapter_path,
                    "agent_type": agent_type,
                    "description": f"DPO adapter for {agent_type} trained on {base_model_id}"
                }
                
                self.config_manager.add_saved_adapter(adapter_config)
                self.config_manager.save_config()
                
                # Update status
                with self._lock:
                    if job_id in self.active_training_jobs:
                        self.active_training_jobs[job_id]["status"] = "completed"
                        self.active_training_jobs[job_id]["end_time"] = datetime.utcnow().isoformat()
                        self.active_training_jobs[job_id]["adapter_id"] = adapter_id
                
                # Call callback if provided
                if callback:
                    callback(job_id, {
                        "type": "status",
                        "value": "completed",
                        "adapter_id": adapter_id
                    })
            else:
                # Training failed
                logger.error(f"Training job {job_id} failed with return code {return_code}")
                
                # Update status
                with self._lock:
                    if job_id in self.active_training_jobs:
                        self.active_training_jobs[job_id]["status"] = "failed"
                        self.active_training_jobs[job_id]["end_time"] = datetime.utcnow().isoformat()
                        self.active_training_jobs[job_id]["error_code"] = return_code
                
                # Call callback if provided
                if callback:
                    callback(job_id, {"type": "status", "value": "failed"})
        
        except Exception as e:
            # Handle exceptions
            logger.error(f"Error in training job {job_id}: {str(e)}")
            
            # Update status
            with self._lock:
                if job_id in self.active_training_jobs:
                    self.active_training_jobs[job_id]["status"] = "failed"
                    self.active_training_jobs[job_id]["end_time"] = datetime.utcnow().isoformat()
                    self.active_training_jobs[job_id]["error"] = str(e)
            
            # Call callback if provided
            if callback:
                callback(job_id, {"type": "error", "value": str(e)})
        
        finally:
            # Clean up temporary data file
            try:
                if os.path.exists(data_path):
                    os.unlink(data_path)
            except Exception as e:
                logger.error(f"Error cleaning up data file {data_path}: {str(e)}")
    
    def cancel_training_job(self, job_id: str) -> bool:
        """Cancel a running training job"""
        with self._lock:
            if job_id not in self.active_training_jobs:
                logger.warning(f"Training job {job_id} not found")
                return False
            
            job = self.active_training_jobs[job_id]
            if job["status"] not in ["running", "starting"]:
                logger.warning(f"Training job {job_id} is not running (status: {job['status']})")
                return False
            
            # In a real implementation, we would need to find and terminate the subprocess
            # This is a simplified implementation
            job["status"] = "cancelling"
        
        logger.info(f"Cancelling training job {job_id}")
        return True
    
    def get_training_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a training job"""
        with self._lock:
            if job_id not in self.active_training_jobs:
                return None
            
            job = self.active_training_jobs[job_id].copy()
            # Remove the thread object from the copy
            if "thread" in job:
                del job["thread"]
            
            return job
    
    def get_training_job_output(self, job_id: str, max_lines: int = 100) -> List[str]:
        """Get the output lines of a training job"""
        with self._lock:
            if job_id not in self.active_training_jobs:
                return []
            
            output = self.active_training_jobs[job_id]["output"]
            return output[-max_lines:] if max_lines > 0 else output
    
    def get_active_training_jobs(self) -> List[Dict[str, Any]]:
        """Get all active training jobs"""
        with self._lock:
            return [
                {k: v for k, v in job.items() if k != "thread"}
                for job in self.active_training_jobs.values()
                if job["status"] in ["starting", "running", "cancelling"]
            ]
    
    def get_training_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about training jobs"""
        with self._lock:
            total_jobs = len(self.active_training_jobs)
            active_jobs = sum(1 for job in self.active_training_jobs.values() 
                               if job["status"] in ["starting", "running", "cancelling"])
            completed_jobs = sum(1 for job in self.active_training_jobs.values() 
                                  if job["status"] == "completed")
            failed_jobs = sum(1 for job in self.active_training_jobs.values() 
                               if job["status"] == "failed")
            
            return {
                "total_jobs": total_jobs,
                "active_jobs": active_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs
            }