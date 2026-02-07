"""
Batch processing engine for TTS dubbing workflows.

This module provides tools for:
- Batch TTS generation from subtitles
- Task queue management
- Progress tracking and persistence
- Audio file organization and export
"""

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
import uuid

from loguru import logger

from .subtitle_processor import SubtitleProcessor, SubtitleEntry


@dataclass
class BatchJobConfig:
    """Configuration for a batch TTS job."""
    job_id: str
    name: str
    subtitle_file: str
    voice_mappings: Dict[str, str]  # character -> voice_id
    output_dir: str
    output_format: str = "wav"
    chunk_length: int = 200
    max_new_tokens: int = 1024
    top_p: float = 0.8
    temperature: float = 0.8
    speed_factor: float = 1.0
    pitch_factor: float = 1.0
    emotion_intensity: float = 1.0
    volume_gain: float = 1.0


@dataclass
class BatchJobStatus:
    """Status of a batch TTS job."""
    job_id: str
    name: str
    status: str  # pending, running, completed, failed, cancelled
    progress: float = 0.0
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    output_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class BatchProcessor:
    """
    Batch TTS processor for dubbing workflows.

    Features:
    - Parse subtitles and group by speaker
    - Generate TTS audio for each segment
    - Organize output files by character
    - Track progress and handle errors
    - Support concurrent processing
    """

    def __init__(
        self,
        max_workers: int = 4,
        tts_function: Optional[Callable] = None,
    ):
        """
        Initialize the batch processor.

        Args:
            max_workers: Maximum number of concurrent TTS jobs
            tts_function: Async function for TTS generation
                         Signature: async def tts(text, voice_id, **kwargs) -> audio_data
        """
        self.max_workers = max_workers
        self.tts_function = tts_function
        self.jobs: Dict[str, BatchJobStatus] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def create_job(
        self,
        name: str,
        subtitle_file: str,
        voice_mappings: Dict[str, str],
        output_dir: str,
        **kwargs,
    ) -> str:
        """
        Create a new batch processing job.

        Args:
            name: Job name
            subtitle_file: Path to SRT subtitle file
            voice_mappings: Character to voice ID mappings
            output_dir: Output directory for generated audio
            **kwargs: Additional TTS parameters

        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create job configuration
        config = BatchJobConfig(
            job_id=job_id,
            name=name,
            subtitle_file=subtitle_file,
            voice_mappings=voice_mappings,
            output_dir=str(output_dir),
            **kwargs,
        )

        # Save job configuration
        config_file = output_dir / f"{job_id}_config.json"
        config_file.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

        # Create job status
        job = BatchJobStatus(
            job_id=job_id,
            name=name,
            status="pending",
            created_at=datetime.utcnow().isoformat(),
        )

        self.jobs[job_id] = job

        logger.info(f"Created batch job: {job_id} - {name}")
        return job_id

    def get_job_status(self, job_id: str) -> Optional[BatchJobStatus]:
        """Get the status of a batch job."""
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[BatchJobStatus]:
        """List all batch jobs."""
        return list(self.jobs.values())

    async def process_job(
        self,
        job_id: str,
        tts_function: Optional[Callable] = None,
    ) -> BatchJobStatus:
        """
        Process a batch TTS job.

        Args:
            job_id: Job ID to process
            tts_function: TTS generation function (uses default if not provided)

        Returns:
            Final job status
        """
        job = self.jobs.get(job_id)

        if not job:
            raise ValueError(f"Job not found: {job_id}")

        if job.status == "running":
            raise ValueError(f"Job already running: {job_id}")

        if job.status in ["completed", "failed", "cancelled"]:
            raise ValueError(f"Job already {job.status}: {job_id}")

        # Load job configuration
        job_dir = Path(job.output_path or ".")
        config_file = job_dir / f"{job_id}_config.json"

        if not config_file.exists():
            job.status = "failed"
            job.error_message = "Configuration file not found"
            return job

        config = json.loads(config_file.read_text())

        # Use provided TTS function or default
        tts_fn = tts_function or self.tts_function

        if not tts_fn:
            job.status = "failed"
            job.error_message = "No TTS function provided"
            return job

        # Update job status
        job.status = "running"
        job.started_at = datetime.utcnow().isoformat()

        try:
            # Parse subtitle file
            processor = SubtitleProcessor()
            subtitle_path = Path(config["subtitle_file"])

            if subtitle_path.suffix == ".srt":
                processor.parse_srt(subtitle_path)
            else:
                processor.parse_ass(subtitle_path)

            # Detect and map characters
            processor.detect_characters()
            processor.set_voice_mappings(config["voice_mappings"])
            processor.apply_voice_mappings()

            # Get segments for TTS
            segments = processor.get_batch_segments(group_by_speaker=True)

            job.total_items = len(segments)
            job.completed_items = 0
            job.failed_items = 0

            # Create output directories for each character
            char_dirs = {}
            for segment in segments:
                speaker = segment.get("speaker", "unknown")
                if speaker and speaker not in char_dirs:
                    char_dir = Path(config["output_dir"]) / speaker
                    char_dir.mkdir(parents=True, exist_ok=True)
                    char_dirs[speaker] = char_dir

            # Process segments concurrently
            semaphore = asyncio.Semaphore(self.max_workers)

            async def process_segment(segment: Dict[str, Any], index: int):
                """Process a single subtitle segment."""
                nonlocal job

                async with semaphore:
                    try:
                        text = segment["text"]
                        voice_id = segment.get("voice_id")
                        speaker = segment.get("speaker", "unknown")

                        # Generate TTS
                        audio_data = await tts_fn(
                            text=text,
                            voice_id=voice_id,
                            format=config.get("output_format", "wav"),
                            chunk_length=config.get("chunk_length", 200),
                            max_new_tokens=config.get("max_new_tokens", 1024),
                            top_p=config.get("top_p", 0.8),
                            temperature=config.get("temperature", 0.8),
                            speed_factor=config.get("speed_factor", 1.0),
                            pitch_factor=config.get("pitch_factor", 1.0),
                            emotion_intensity=config.get("emotion_intensity", 1.0),
                            volume_gain=config.get("volume_gain", 1.0),
                        )

                        # Save audio file
                        if speaker and speaker in char_dirs:
                            output_file = char_dirs[speaker] / f"line_{index:04d}.wav"

                            # Write audio data
                            # Assuming audio_data is bytes or numpy array
                            import soundfile as sf
                            import numpy as np

                            if isinstance(audio_data, bytes):
                                # If bytes, need to decode
                                audio_array = np.frombuffer(audio_data, dtype=np.float16)
                            else:
                                audio_array = audio_data

                            sf.write(output_file, audio_array, 24000)

                        # Update progress
                        job.completed_items += 1
                        job.progress = job.completed_items / job.total_items

                        logger.debug(f"Processed segment {index + 1}/{job.total_items}")

                    except Exception as e:
                        logger.error(f"Failed to process segment {index}: {e}")
                        job.failed_items += 1
                        raise

            # Process all segments
            tasks = [process_segment(seg, i) for i, seg in enumerate(segments)]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Create manifest file
            manifest = {
                "job_id": job_id,
                "name": config["name"],
                "total_segments": len(segments),
                "completed_segments": job.completed_items,
                "failed_segments": job.failed_items,
                "characters": list(char_dirs.keys()),
                "segments": segments,
                "created_at": job.created_at,
                "completed_at": datetime.utcnow().isoformat(),
            }

            manifest_file = Path(config["output_dir"]) / f"{job_id}_manifest.json"
            manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

            # Update job status
            if job.failed_items > 0:
                job.status = "partial"
                job.error_message = f"{job.failed_items} segments failed"
            else:
                job.status = "completed"

            job.completed_at = datetime.utcnow().isoformat()
            job.output_path = config["output_dir"]

            logger.info(f"Completed batch job: {job_id} - {job.completed_items}/{job.total_items} segments")

        except Exception as e:
            logger.error(f"Error processing batch job {job_id}: {e}", exc_info=True)
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow().isoformat()

        return job

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running batch job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancelled, False otherwise
        """
        job = self.jobs.get(job_id)

        if not job:
            return False

        if job.status not in ["pending", "running"]:
            return False

        job.status = "cancelled"
        job.completed_at = datetime.utcnow().isoformat()

        logger.info(f"Cancelled batch job: {job_id}")
        return True

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a batch job and its files.

        Args:
            job_id: Job ID to delete

        Returns:
            True if deleted, False otherwise
        """
        job = self.jobs.get(job_id)

        if not job:
            return False

        # Cannot delete running jobs
        if job.status == "running":
            return False

        # Delete job files if they exist
        if job.output_path:
            job_dir = Path(job.output_path)
            if job_dir.exists():
                # Delete job-specific files
                for pattern in [f"{job_id}_*", f"{job_id}.*"]:
                    for file in job_dir.glob(pattern):
                        file.unlink()

        # Remove from jobs dict
        del self.jobs[job_id]

        logger.info(f"Deleted batch job: {job_id}")
        return True

    def save_state(self, state_file: str | Path) -> None:
        """
        Save job states to file for persistence.

        Args:
            state_file: Path to state file
        """
        state_file = Path(state_file)
        state_file.parent.mkdir(parents=True, exist_ok=True)

        state_data = {
            "jobs": {job_id: job.to_dict() for job_id, job in self.jobs.items()},
            "saved_at": datetime.utcnow().isoformat(),
        }

        state_file.write_text(json.dumps(state_data, indent=2), encoding="utf-8")
        logger.info(f"Saved state to: {state_file}")

    def load_state(self, state_file: str | Path) -> None:
        """
        Load job states from file.

        Args:
            state_file: Path to state file
        """
        state_file = Path(state_file)

        if not state_file.exists():
            logger.warning(f"State file not found: {state_file}")
            return

        state_data = json.loads(state_file.read_text(encoding="utf-8"))

        for job_id, job_dict in state_data.get("jobs", {}).items():
            self.jobs[job_id] = BatchJobStatus(**job_dict)

        logger.info(f"Loaded state from: {state_file} - {len(self.jobs)} jobs")


# Mock TTS function for testing
async def mock_tts(
    text: str,
    voice_id: Optional[str] = None,
    **kwargs,
) -> bytes:
    """Mock TTS function for testing."""
    import numpy as np

    # Generate 1 second of silence per character
    duration = max(1.0, len(text) / 10.0)
    sample_rate = 24000
    samples = int(duration * sample_rate)

    # Generate simple tone
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples)) * 0.1

    return audio.astype(np.float16).tobytes()


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch TTS processor for dubbing"
    )
    parser.add_argument("subtitle", help="Input SRT subtitle file")
    parser.add_argument("output", help="Output directory")
    parser.add_argument(
        "--voices", nargs="+", default=[],
        help="Voice mappings (character:voice_id pairs)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Maximum concurrent TTS jobs"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run with mock TTS for testing"
    )

    args = parser.parse_args()

    # Parse voice mappings
    voice_mappings = {}
    for mapping in args.voices:
        if ":" in mapping:
            char, voice_id = mapping.split(":", 1)
            voice_mappings[char] = voice_id

    # Create processor
    processor = BatchProcessor(max_workers=args.workers)

    if args.test:
        processor.tts_function = mock_tts

    # Create job
    job_id = processor.create_job(
        name=f"Batch_{Path(args.subtitle).stem}",
        subtitle_file=args.subtitle,
        voice_mappings=voice_mappings,
        output_dir=args.output,
    )

    # Process job
    async def run():
        status = await processor.process_job(job_id)
        print(f"Job completed: {status.status}")
        print(f"Progress: {status.completed_items}/{status.total_items}")

    asyncio.run(run())

    # Save state
    processor.save_state(Path(args.output) / "batch_state.json")
