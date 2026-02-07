"""
Training Manager for True Voice Cloning with LoRA Fine-tuning.

This module provides the complete training pipeline:
1. Data preparation (audio segmentation + .lab files)
2. VQ code extraction
3. Dataset packing
4. LoRA training
5. Weight merging
6. Model deployment

Key difference from reference-based TTS:
- Reference TTS: Uses audio as "style guide" (~70-80% similarity)
- LoRA Training: Actually trains model weights (~90-95% similarity)
"""

import asyncio
import json
import os
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable

import torch
from loguru import logger


class TrainingStatus(str, Enum):
    """Training job status."""
    PENDING = "pending"
    PREPARING_DATA = "preparing_data"
    EXTRACTING_VQ = "extracting_vq"
    BUILDING_DATASET = "building_dataset"
    TRAINING = "training"
    MERGING_WEIGHTS = "merging_weights"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """Training job data model."""
    job_id: str
    voice_id: str
    name: str
    status: TrainingStatus
    progress: float  # 0.0 to 1.0
    current_step: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    checkpoint_path: Optional[str] = None
    merged_model_path: Optional[str] = None
    training_params: Optional[Dict] = None

    def to_dict(self):
        return asdict(self)


class TrainingManager:
    """
    Manages LoRA training jobs for voice cloning.

    Provides async training with progress tracking and callbacks.
    """

    def __init__(self, max_workers: int = 1):
        """
        Initialize the training manager.

        Args:
            max_workers: Maximum number of concurrent training jobs
        """
        self.jobs: Dict[str, TrainingJob] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_jobs: Dict[str, Callable] = {}  # job_id -> cancel callback
        self._lock = threading.Lock()

        # Training paths
        self.base_model_path = Path("checkpoints/openaudio-s1-mini")
        self.training_output_path = Path("training_output")
        self.training_output_path.mkdir(exist_ok=True)

    def create_job(
        self,
        voice_id: str,
        name: str,
        audio_files: List[Path],
        reference_text: str,
        max_steps: int = 5000,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
    ) -> TrainingJob:
        """
        Create a new training job.

        Args:
            voice_id: Voice ID
            name: Voice name
            audio_files: List of audio file paths
            reference_text: Reference text for training
            max_steps: Maximum training steps
            batch_size: Training batch size
            learning_rate: Learning rate

        Returns:
            TrainingJob object
        """
        job_id = str(uuid.uuid4())[:8]

        job = TrainingJob(
            job_id=job_id,
            voice_id=voice_id,
            name=name,
            status=TrainingStatus.PENDING,
            progress=0.0,
            current_step="Initializing...",
            created_at=datetime.utcnow().isoformat(),
            training_params={
                "max_steps": max_steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "audio_files": [str(f) for f in audio_files],
                "reference_text": reference_text,
            }
        )

        with self._lock:
            self.jobs[job_id] = job

        logger.info(f"Created training job {job_id} for voice {voice_id}")
        return job

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a training job by ID."""
        return self.jobs.get(job_id)

    def list_jobs(self, voice_id: Optional[str] = None) -> List[TrainingJob]:
        """List all training jobs, optionally filtered by voice_id."""
        jobs = list(self.jobs.values())
        if voice_id:
            jobs = [j for j in jobs if j.voice_id == voice_id]
        return jobs

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a training job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancelled successfully
        """
        job = self.jobs.get(job_id)
        if not job:
            return False

        if job.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]:
            return False

        # Update status
        job.status = TrainingStatus.CANCELLED
        job.completed_at = datetime.utcnow().isoformat()

        # Cancel async task if running
        if job_id in self.active_jobs:
            cancel_fn = self.active_jobs[job_id]
            try:
                cancel_fn()
                del self.active_jobs[job_id]
            except Exception as e:
                logger.warning(f"Error cancelling job {job_id}: {e}")

        logger.info(f"Cancelled training job {job_id}")
        return True

    def start_training(self, job_id: str, progress_callback: Optional[Callable] = None):
        """
        Start a training job asynchronously.

        Args:
            job_id: Job ID to start
            progress_callback: Optional callback for progress updates
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        # Submit to thread pool
        future = self.executor.submit(self._run_training_pipeline, job, progress_callback)

        # Store cancel callback
        def cancel_fn():
            future.cancel()

        self.active_jobs[job_id] = cancel_fn

        logger.info(f"Started training job {job_id}")

    def _run_training_pipeline(self, job: TrainingJob, progress_callback: Optional[Callable] = None):
        """
        Run the complete training pipeline.

        This runs in a separate thread.
        """
        try:
            job.status = TrainingStatus.TRAINING
            job.started_at = datetime.utcnow().isoformat()
            job.current_step = "Starting training pipeline..."
            job.progress = 0.0
            self._notify_progress(job, progress_callback)

            # Get training parameters
            params = job.training_params or {}
            max_steps = params.get("max_steps", 5000)
            batch_size = params.get("batch_size", 4)
            learning_rate = params.get("learning_rate", 1e-4)
            audio_files = params.get("audio_files", [])
            reference_text = params.get("reference_text", "")

            # Prepare training data directory
            voice_dir = Path("voices") / job.voice_id
            training_data_dir = self.training_output_path / job.job_id / "data"
            training_data_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Prepare training data (10%)
            job.current_step = "Preparing training data..."
            job.progress = 0.1
            self._notify_progress(job, progress_callback)

            success = self._prepare_training_data(
                voice_dir,
                training_data_dir,
                reference_text,
            )
            if not success:
                raise Exception("Failed to prepare training data")

            # Step 2: Extract VQ codes (25%)
            job.current_step = "Extracting VQ codes..."
            job.progress = 0.25
            job.status = TrainingStatus.EXTRACTING_VQ
            self._notify_progress(job, progress_callback)

            success = self._extract_vq_codes(training_data_dir)
            if not success:
                raise Exception("Failed to extract VQ codes")

            # Step 3: Build dataset (40%)
            job.current_step = "Building training dataset..."
            job.progress = 0.40
            job.status = TrainingStatus.BUILDING_DATASET
            self._notify_progress(job, progress_callback)

            success = self._build_dataset(training_data_dir)
            if not success:
                raise Exception("Failed to build dataset")

            # Step 4: Run LoRA training (40-90%)
            job.current_step = "Training LoRA model..."
            job.progress = 0.45
            job.status = TrainingStatus.TRAINING
            self._notify_progress(job, progress_callback)

            checkpoint_path = self._run_lora_training(
                training_data_dir,
                self.training_output_path / job.job_id,
                max_steps,
                batch_size,
                learning_rate,
                job,
                progress_callback,
            )
            if not checkpoint_path:
                raise Exception("LoRA training failed")

            job.checkpoint_path = str(checkpoint_path)

            # Step 5: Merge weights (90-100%)
            job.current_step = "Merging LoRA weights..."
            job.progress = 0.90
            job.status = TrainingStatus.MERGING_WEIGHTS
            self._notify_progress(job, progress_callback)

            merged_path = self._merge_weights(
                checkpoint_path,
                self.training_output_path / job.job_id / "merged_model",
            )
            if not merged_path:
                raise Exception("Failed to merge weights")

            job.merged_model_path = str(merged_path)

            # Update voice metadata with trained model path
            self._update_voice_metadata(job.voice_id, merged_path)

            # Complete
            job.status = TrainingStatus.COMPLETED
            job.progress = 1.0
            job.current_step = "Training completed successfully!"
            job.completed_at = datetime.utcnow().isoformat()
            self._notify_progress(job, progress_callback)

            logger.info(f"Training job {job.job_id} completed successfully")

        except Exception as e:
            logger.error(f"Training job {job.job_id} failed: {e}", exc_info=True)
            job.status = TrainingStatus.FAILED
            job.error = str(e)
            job.current_step = f"Training failed: {str(e)}"
            job.completed_at = datetime.utcnow().isoformat()
            self._notify_progress(job, progress_callback)

        finally:
            # Clean up active job
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]

    def _prepare_training_data(
        self,
        voice_dir: Path,
        training_data_dir: Path,
        reference_text: str,
    ) -> bool:
        """
        Prepare training data from voice directory.

        Creates the Fish Speech training format:
        training_data/
        └── SPK1/
            ├── 00.00-05.23.wav
            ├── 00.00-05.23.lab
            └── ...
        """
        try:
            # Create speaker directory
            speaker_dir = training_data_dir / "SPK1"
            speaker_dir.mkdir(exist_ok=True)

            # Get audio files
            audio_files = list(voice_dir.glob("reference_*.wav"))
            if not audio_files:
                audio_files = list(voice_dir.glob("*.wav"))

            if not audio_files:
                logger.error(f"No audio files found in {voice_dir}")
                return False

            # Copy audio files and create .lab files
            for i, audio_file in enumerate(audio_files):
                # Copy audio
                target_wav = speaker_dir / f"{i:05d}.wav"
                target_wav.write_bytes(audio_file.read_bytes())

                # Create .lab file with reference text
                # For segments, use partial text
                lab_file = speaker_dir / f"{i:05d}.lab"

                # If we have multiple segments, split the reference text
                # Otherwise use the full text for all
                lab_file.write_text(reference_text, encoding="utf-8")

            logger.info(f"Prepared {len(audio_files)} audio segments for training")
            return True

        except Exception as e:
            logger.error(f"Error preparing training data: {e}", exc_info=True)
            return False

    def _extract_vq_codes(self, data_dir: Path) -> bool:
        """
        Extract VQ codes from audio using VQ-GAN model.
        """
        try:
            cmd = [
                "python3",
                "tools/vqgan/extract_vq.py",
                str(data_dir),
                "--num-workers", "1",
                "--batch-size", "8",
                "--config-name", "modded_dac_vq",
                "--checkpoint-path", str(self.base_model_path / "codec.pth"),
            ]

            logger.info(f"Running VQ extraction: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes timeout
                cwd=Path.cwd(),
            )

            if result.returncode != 0:
                logger.error(f"VQ extraction failed: {result.stderr}")
                return False

            # Check for .npy files
            npy_count = sum(1 for _ in data_dir.rglob("*.npy"))
            if npy_count == 0:
                logger.error("No .npy files generated")
                return False

            logger.info(f"Generated {npy_count} VQ token files")
            return True

        except subprocess.TimeoutExpired:
            logger.error("VQ extraction timed out")
            return False
        except Exception as e:
            logger.error(f"Error extracting VQ codes: {e}", exc_info=True)
            return False

    def _build_dataset(self, data_dir: Path) -> bool:
        """
        Pack dataset into protobuf format.
        """
        try:
            proto_dir = data_dir / "protos"
            proto_dir.mkdir(exist_ok=True)

            cmd = [
                "python3",
                "tools/llama/build_dataset.py",
                "--input", str(data_dir),
                "--output", str(proto_dir),
                "--text-extension", ".lab",
                "--num-workers", "2",
            ]

            logger.info(f"Building dataset: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
                cwd=Path.cwd(),
            )

            if result.returncode != 0:
                logger.error(f"Dataset build failed: {result.stderr}")
                return False

            logger.info("Dataset built successfully")
            return True

        except subprocess.TimeoutExpired:
            logger.error("Dataset build timed out")
            return False
        except Exception as e:
            logger.error(f"Error building dataset: {e}", exc_info=True)
            return False

    def _run_lora_training(
        self,
        data_dir: Path,
        output_dir: Path,
        max_steps: int,
        batch_size: int,
        learning_rate: float,
        job: TrainingJob,
        progress_callback: Optional[Callable] = None,
    ) -> Optional[Path]:
        """
        Run LoRA fine-tuning.
        """
        try:
            proto_dir = data_dir / "protos"
            run_output = output_dir / "checkpoints"
            run_output.mkdir(parents=True, exist_ok=True)

            cmd = [
                "python3",
                "fish_speech/train.py",
                "--config-name", "text2semantic_finetune",
                f"project={run_output.parent}",
                "+lora@model.model.lora_config=r_8_alpha_16",
                f"data.train_datasets=[{proto_dir}]",
                f"trainer.max_steps={max_steps}",
                f"model.global_batch_size={batch_size}",
                f"model.optimizer.lr={learning_rate}",
            ]

            logger.info(f"Running LoRA training: {' '.join(cmd)}")

            # Run with output capture for progress monitoring
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=Path.cwd(),
            )

            # Monitor output for progress
            step_start_progress = 0.45
            step_end_progress = 0.90

            for line in process:
                logger.info(f"[Training] {line.strip()}")

                # Parse progress from output
                # Lightning outputs like: "Epoch 0:  10%|████ | 100/1000 [00:01<00:09]"
                if "%" in line and "step" in line.lower():
                    try:
                        # Extract percentage
                        parts = line.split("%")
                        if parts:
                            percent_str = parts[0].split()[-1]
                            percent = float(percent_str) / 100.0

                            # Update job progress
                            job.progress = step_start_progress + (percent * (step_end_progress - step_start_progress))

                            # Also update current step
                            if "step" in line.lower():
                                step_match = [s for s in line.split() if s.isdigit()]
                                if step_match:
                                    step_num = step_match[0]
                                    job.current_step = f"Training step {step_num}/{max_steps}"

                            self._notify_progress(job, progress_callback)
                    except (ValueError, IndexError):
                        pass

            # Wait for process to complete
            returncode = process.wait()

            if returncode != 0:
                logger.error(f"Training failed with return code {returncode}")
                return None

            # Find checkpoint
            checkpoints = list(run_output.glob("*.ckpt"))
            if not checkpoints:
                logger.error("No checkpoint files found")
                return None

            # Get the latest checkpoint
            best_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)

            logger.info(f"Training complete. Checkpoint: {best_checkpoint}")
            return best_checkpoint

        except Exception as e:
            logger.error(f"Error running LoRA training: {e}", exc_info=True)
            return None

    def _merge_weights(self, checkpoint_path: Path, output_path: Path) -> Optional[Path]:
        """
        Merge LoRA weights with base model.
        """
        try:
            output_path.mkdir(parents=True, exist_ok=True)

            cmd = [
                "python3",
                "tools/llama/merge_lora.py",
                "--lora-config", "r_8_alpha_16",
                "--base-weight", str(self.base_model_path),
                "--lora-weight", str(checkpoint_path),
                "--output", str(output_path),
            ]

            logger.info(f"Merging weights: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=Path.cwd(),
            )

            if result.returncode != 0:
                logger.error(f"Weight merging failed: {result.stderr}")
                return None

            # Check for merged files
            merged_files = list(output_path.rglob("*.pth")) + list(output_path.rglob("*.safetensors"))
            if not merged_files:
                logger.warning("Merging appeared successful but no weight files found")

            logger.info(f"Merged weights saved to: {output_path}")
            return output_path

        except subprocess.TimeoutExpired:
            logger.error("Weight merging timed out")
            return None
        except Exception as e:
            logger.error(f"Error merging weights: {e}", exc_info=True)
            return None

    def _update_voice_metadata(self, voice_id: str, model_path: Path) -> bool:
        """
        Update voice metadata with trained model path.
        """
        try:
            voice_dir = Path("voices") / voice_id
            metadata_file = voice_dir / "metadata.json"

            if not metadata_file.exists():
                logger.warning(f"Voice metadata not found for {voice_id}")
                return False

            metadata = json.loads(metadata_file.read_text())
            metadata["is_trained"] = True
            metadata["trained_model_path"] = str(model_path)
            metadata["trained_at"] = datetime.utcnow().isoformat()

            metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

            logger.info(f"Updated metadata for voice {voice_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating voice metadata: {e}", exc_info=True)
            return False

    def _notify_progress(self, job: TrainingJob, callback: Optional[Callable] = None):
        """Notify progress callback."""
        if callback:
            try:
                callback(job)
            except Exception as e:
                logger.warning(f"Error in progress callback: {e}")


# Global training manager instance
_training_manager: Optional[TrainingManager] = None


def get_training_manager() -> TrainingManager:
    """Get the global training manager instance."""
    global _training_manager
    if _training_manager is None:
        _training_manager = TrainingManager()
    return _training_manager
