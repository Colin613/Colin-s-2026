"""
Automated LoRA fine-tuning script for Yanbian Korean voice cloning.

This script automates the complete fine-tuning workflow:
1. Data validation and preprocessing
2. VQ token extraction
3. Dataset packing (protobuf)
4. LoRA training
5. Weight merging
6. Model validation

Usage:
    python tools/yanbian_finetune.py --data data/yanbian_voice --output checkpoints/yanbian_voice
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pyrootutils
from loguru import logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class YanbianFinetunePipeline:
    """
    Automated fine-tuning pipeline for Yanbian Korean voices.

    Features:
    - Data validation
    - VQ token extraction
    - Dataset preparation
    - LoRA training
    - Weight merging
    - Progress tracking
    """

    def __init__(
        self,
        data_path: str | Path,
        output_path: str | Path,
        base_model: str = "checkpoints/openaudio-s1-mini",
        lora_config: str = "r_8_alpha_16",
        max_steps: int = 5000,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        device: str = "cuda",
    ):
        """
        Initialize the fine-tuning pipeline.

        Args:
            data_path: Path to training data (speaker directories)
            output_path: Path for output checkpoints
            base_model: Path to base model weights
            lora_config: LoRA configuration name
            max_steps: Maximum training steps
            batch_size: Training batch size
            learning_rate: Learning rate
            device: Device to use (cuda/cpu)
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.base_model = Path(base_model)
        self.lora_config = lora_config
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_path / f"finetune_{timestamp}.log"

        # Add file logger
        logger.add(str(self.log_file), rotation="100 MB")

        # Pipeline state
        self.state = {
            "data_path": str(self.data_path),
            "output_path": str(self.output_path),
            "base_model": str(self.base_model),
            "lora_config": lora_config,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": device,
            "started_at": datetime.utcnow().isoformat(),
            "steps": {},
        }

    def validate_data(self) -> bool:
        """
        Validate training data directory structure.

        Expected structure:
        data/
        ├── SPK1/
        │   ├── 00.00-05.23.wav
        │   ├── 00.00-05.23.lab
        │   └── ...
        └── SPK2/
            └── ...

        Returns:
            True if valid, False otherwise
        """
        logger.info("Step 1: Validating training data...")

        if not self.data_path.exists():
            logger.error(f"Data path does not exist: {self.data_path}")
            return False

        # Check for speaker directories
        speaker_dirs = [d for d in self.data_path.iterdir() if d.is_dir()]

        if not speaker_dirs:
            logger.error(f"No speaker directories found in: {self.data_path}")
            return False

        logger.info(f"Found {len(speaker_dirs)} speaker directories")

        # Validate each speaker directory
        valid_speakers = []
        total_segments = 0

        for speaker_dir in speaker_dirs:
            # Check for .wav and .lab files
            wav_files = list(speaker_dir.glob("*.wav"))
            lab_files = list(speaker_dir.glob("*.lab"))

            # Check for paired files
            paired_count = 0
            for wav_file in wav_files:
                lab_file = speaker_dir / f"{wav_file.stem}.lab"
                if lab_file.exists() and lab_file.stat().st_size > 0:
                    paired_count += 1

            if paired_count > 0:
                valid_speakers.append(speaker_dir.name)
                total_segments += paired_count
                logger.info(f"  {speaker_dir.name}: {paired_count} paired segments")

        if not valid_speakers:
            logger.error("No valid speaker directories found")
            return False

        logger.info(f"Validation complete: {len(valid_speakers)} speakers, {total_segments} segments")

        self.state["steps"]["validation"] = {
            "valid_speakers": len(valid_speakers),
            "total_segments": total_segments,
            "speakers": valid_speakers,
        }

        return True

    def extract_vq_tokens(self) -> bool:
        """
        Extract VQ tokens from audio using VQ-GAN model.

        Returns:
            True if successful, False otherwise
        """
        logger.info("Step 2: Extracting VQ tokens...")

        # Build command
        cmd = [
            sys.executable,
            "tools/vqgan/extract_vq.py",
            str(self.data_path),
            "--num-workers", "1",
            "--batch-size", "16",
            "--config-name", "modded_dac_vq",
            "--checkpoint-path", str(self.base_model / "codec.pth"),
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )

            logger.info(result.stdout)

            # Check for .npy files
            npy_count = sum(1 for _ in self.data_path.rglob("*.npy"))

            if npy_count == 0:
                logger.error("No .npy files generated")
                return False

            logger.info(f"Generated {npy_count} VQ token files")

            self.state["steps"]["vq_extraction"] = {
                "success": True,
                "npy_files": npy_count,
            }

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"VQ extraction failed: {e.stderr}")
            self.state["steps"]["vq_extraction"] = {"success": False, "error": str(e)}
            return False

    def build_dataset(self) -> bool:
        """
        Pack dataset into protobuf format.

        Returns:
            True if successful, False otherwise
        """
        logger.info("Step 3: Building dataset...")

        proto_dir = self.data_path / "protos"
        proto_dir.mkdir(exist_ok=True)

        # Build command
        cmd = [
            sys.executable,
            "tools/llama/build_dataset.py",
            "--input", str(self.data_path),
            "--output", str(proto_dir),
            "--text-extension", ".lab",
            "--num-workers", "4",
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )

            logger.info(result.stdout)

            # Check for proto files
            proto_files = list(proto_dir.glob("*.proto")) + list(proto_dir.glob("*.json"))

            if not proto_files:
                logger.error("No proto files generated")
                return False

            logger.info(f"Built dataset with {len(proto_files)} proto files")

            self.state["steps"]["dataset_build"] = {
                "success": True,
                "proto_files": len(proto_files),
                "proto_dir": str(proto_dir),
            }

            self.proto_dir = proto_dir
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Dataset build failed: {e.stderr}")
            self.state["steps"]["dataset_build"] = {"success": False, "error": str(e)}
            return False

    def run_training(self) -> bool:
        """
        Run LoRA fine-tuning.

        Returns:
            True if successful, False otherwise
        """
        logger.info("Step 4: Running LoRA fine-tuning...")

        # Create output directory for this run
        run_name = f"yanbian_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_output = self.output_path / run_name
        run_output.mkdir(exist_ok=True)

        # Build command
        cmd = [
            sys.executable,
            "fish_speech/train.py",
            "--config-name", "text2semantic_finetune",
            f"project={run_output}",
            f"+lora@model.model.lora_config={self.lora_config}",
        ]

        # Override training parameters
        overrides = [
            f"data.train_datasets=[{self.proto_dir}]",
            f"trainer.max_steps={self.max_steps}",
            f"model.global_batch_size={self.batch_size}",
            f"model.optimizer.lr={self.learning_rate}",
        ]

        cmd.extend(overrides)

        logger.info(f"Running: {' '.join(cmd)}")
        logger.info(f"Output directory: {run_output}")

        try:
            # Run training
            result = subprocess.run(
                cmd,
                check=False,  # Allow non-zero exit for training issues
                capture_output=True,
                text=True,
            )

            # Log output
            logger.info(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)

            # Check for checkpoints
            checkpoint_dir = run_output / "checkpoints"
            if not checkpoint_dir.exists():
                logger.error("No checkpoint directory created")
                return False

            checkpoints = list(checkpoint_dir.glob("*.ckpt"))

            if not checkpoints:
                logger.error("No checkpoint files found")
                return False

            # Find best checkpoint
            best_checkpoint = max(checkpoints, key=lambda p: p.stat().st_size)

            logger.info(f"Training complete. Best checkpoint: {best_checkpoint.name}")

            self.state["steps"]["training"] = {
                "success": True,
                "checkpoint": str(best_checkpoint),
                "run_output": str(run_output),
            }

            self.checkpoint_path = best_checkpoint
            self.run_output = run_output
            return True

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.state["steps"]["training"] = {"success": False, "error": str(e)}
            return False

    def merge_weights(self) -> bool:
        """
        Merge LoRA weights with base model.

        Returns:
            True if successful, False otherwise
        """
        logger.info("Step 5: Merging LoRA weights...")

        if not hasattr(self, "checkpoint_path"):
            logger.error("No checkpoint available for merging")
            return False

        # Output path for merged weights
        merged_output = self.output_path / "merged_model"

        # Build command
        cmd = [
            sys.executable,
            "tools/llama/merge_lora.py",
            "--lora-config", self.lora_config,
            "--base-weight", str(self.base_model),
            "--lora-weight", str(self.checkpoint_path),
            "--output", str(merged_output),
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )

            logger.info(result.stdout)

            # Check for merged weights
            merged_files = list(merged_output.rglob("*.pth")) + list(merged_output.rglob("*.safetensors"))

            if not merged_files:
                logger.warning("Merging appeared successful but no weight files found")

            logger.info(f"Merged weights saved to: {merged_output}")

            self.state["steps"]["merge"] = {
                "success": True,
                "output": str(merged_output),
            }

            self.merged_output = merged_output
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Weight merging failed: {e.stderr}")
            self.state["steps"]["merge"] = {"success": False, "error": str(e)}
            return False

    def save_state(self) -> None:
        """Save pipeline state to JSON file."""
        self.state["completed_at"] = datetime.utcnow().isoformat()

        state_file = self.output_path / "pipeline_state.json"

        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved pipeline state to: {state_file}")

    def run(self, skip_to: Optional[str] = None) -> bool:
        """
        Run the complete fine-tuning pipeline.

        Args:
            skip_to: Skip to this step (validation, vq_extraction, dataset_build, training, merge)

        Returns:
            True if all steps successful, False otherwise
        """
        steps = ["validation", "vq_extraction", "dataset_build", "training", "merge"]

        if skip_to:
            try:
                start_idx = steps.index(skip_to)
            except ValueError:
                logger.error(f"Invalid step name: {skip_to}")
                return False
            steps = steps[start_idx:]

        logger.info(f"Starting fine-tuning pipeline: {len(steps)} steps")

        for step in steps:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running step: {step}")
            logger.info(f"{'='*60}")

            success = False

            if step == "validation":
                success = self.validate_data()
            elif step == "vq_extraction":
                success = self.extract_vq_tokens()
            elif step == "dataset_build":
                success = self.build_dataset()
            elif step == "training":
                success = self.run_training()
            elif step == "merge":
                success = self.merge_weights()

            if not success:
                logger.error(f"Step '{step}' failed. Stopping pipeline.")
                self.save_state()
                return False

            # Save state after each step
            self.save_state()

        logger.info(f"\n{'='*60}")
        logger.info("Pipeline completed successfully!")
        logger.info(f"{'='*60}")

        if hasattr(self, "merged_output"):
            logger.info(f"Final model: {self.merged_output}")

        self.save_state()
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Automated LoRA fine-tuning for Yanbian Korean voices"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to training data directory"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to output directory"
    )
    parser.add_argument(
        "--base-model", default="checkpoints/openaudio-s1-mini",
        help="Path to base model"
    )
    parser.add_argument(
        "--lora-config", default="r_8_alpha_16",
        help="LoRA configuration name"
    )
    parser.add_argument(
        "--max-steps", type=int, default=5000,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--skip-to",
        choices=["validation", "vq_extraction", "dataset_build", "training", "merge"],
        help="Skip to this step"
    )

    args = parser.parse_args()

    pipeline = YanbianFinetunePipeline(
        data_path=args.data,
        output_path=args.output,
        base_model=args.base_model,
        lora_config=args.lora_config,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )

    success = pipeline.run(skip_to=args.skip_to)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
