"""
Long audio segmentation tool for voice cloning training.

This module provides utilities for segmenting long audio files (30+ minutes)
into smaller chunks suitable for training data preparation.
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import librosa
import numpy as np
from loguru import logger

from .audio_processor import AudioProcessor, TARGET_SAMPLE_RATE


class AudioSegmenter:
    """
    Advanced audio segmentation for long-form content.

    Features:
    - Intelligent silence-based segmentation
    - Speaker change detection (for multi-speaker content)
    - Segment duration constraints
    - Metadata preservation
    """

    def __init__(
        self,
        min_segment_duration: float = 5.0,
        max_segment_duration: float = 15.0,
        target_segment_duration: float = 10.0,
        silence_threshold: float = 0.01,
        min_silence_duration: float = 0.5,
    ):
        """
        Initialize the segmenter.

        Args:
            min_segment_duration: Minimum segment duration in seconds
            max_segment_duration: Maximum segment duration in seconds
            target_segment_duration: Target segment duration in seconds
            silence_threshold: RMS threshold for silence detection
            min_silence_duration: Minimum silence duration for split point
        """
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        self.target_segment_duration = target_segment_duration
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration
        self.audio_processor = AudioProcessor()

    def detect_speaker_changes(
        self,
        audio: np.ndarray,
        sr: int,
        window_size: float = 3.0,
        hop_size: float = 0.5,
    ) -> List[int]:
        """
        Detect potential speaker change points using MFCC similarity.

        This is a simple approach; for production use, consider using
        dedicated speaker diarization models like pyannote.audio.

        Args:
            audio: Audio data array
            sr: Sample rate
            window_size: Analysis window size in seconds
            hop_size: Hop size between windows in seconds

        Returns:
            List of sample indices where speaker changes may occur
        """
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)

        # Extract MFCCs
        mfccs = []
        for i in range(0, len(audio) - window_samples, hop_samples):
            window = audio[i : i + window_samples]
            mfcc = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13)
            mfccs.append(mfcc.mean(axis=1))

        if len(mfccs) < 2:
            return []

        mfccs = np.array(mfccs)

        # Calculate MFCC distance between consecutive windows
        change_points = []
        for i in range(len(mfccs) - 1):
            dist = np.linalg.norm(mfccs[i] - mfccs[i + 1])

            # Threshold for speaker change (empirical)
            if dist > 15.0:  # Adjust based on your data
                sample_idx = (i + 1) * hop_samples
                change_points.append(sample_idx)

        logger.info(f"Detected {len(change_points)} potential speaker changes")
        return change_points

    def find_optimal_split_points(
        self,
        audio: np.ndarray,
        sr: int,
        speaker_changes: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Find optimal split points combining silence and speaker changes.

        Args:
            audio: Audio data array
            sr: Sample rate
            speaker_changes: Optional list of speaker change points

        Returns:
            List of sample indices for splitting
        """
        # Get silence-based split points
        silent_segments = self.audio_processor.detect_silence(
            audio,
            sr,
            silence_threshold=self.silence_threshold,
            min_silence_duration=self.min_silence_duration,
        )

        # Convert silence segments to split points (midpoints)
        silence_splits = [
            int((start + end) / 2 * sr) for start, end in silent_segments
        ]

        # Combine with speaker change points
        all_splits = set(silence_splits)
        if speaker_changes:
            all_splits.update(speaker_changes)

        # Sort and add boundaries
        split_points = sorted([0] + list(all_splits) + [len(audio)])

        return split_points

    def create_segments(
        self,
        audio: np.ndarray,
        sr: int,
        split_points: List[int],
    ) -> List[Tuple[int, int, str]]:
        """
        Create segments from split points, respecting duration constraints.

        Args:
            audio: Audio data array
            sr: Sample rate
            split_points: List of sample indices for splitting

        Returns:
            List of (start_sample, end_sample, time_range) tuples
        """
        segments = []

        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            duration = (end - start) / sr

            if self.min_segment_duration <= duration <= self.max_segment_duration:
                time_range = f"{start/sr:.2f}-{end/sr:.2f}"
                segments.append((start, end, time_range))
            elif duration > self.max_segment_duration:
                # Split long segments at target duration
                num_subsegments = int(duration // self.target_segment_duration) + 1
                subsegment_length = (end - start) // num_subsegments

                for j in range(num_subsegments):
                    sub_start = start + j * subsegment_length
                    sub_end = min(start + (j + 1) * subsegment_length, end)
                    sub_duration = (sub_end - sub_start) / sr

                    if sub_duration >= self.min_segment_duration:
                        time_range = f"{sub_start/sr:.2f}-{sub_end/sr:.2f}"
                        segments.append((sub_start, sub_end, time_range))

        return segments

    def segment_long_audio(
        self,
        audio_path: str | Path,
        output_dir: str | Path,
        speaker_id: str = "SPK1",
        detect_speakers: bool = False,
        save_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Segment a long audio file for training data.

        Args:
            audio_path: Path to input audio file
            output_dir: Output directory path
            speaker_id: Speaker identifier for naming
            detect_speakers: Whether to detect speaker changes
            save_metadata: Whether to save segment metadata JSON

        Returns:
            List of segment info dictionaries
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Segmenting long audio: {audio_path}")

        # Load audio
        audio, sr = self.audio_processor.load_audio(audio_path)

        # Detect speaker changes if requested
        speaker_changes = None
        if detect_speakers:
            speaker_changes = self.detect_speaker_changes(audio, sr)

        # Find split points
        split_points = self.find_optimal_split_points(audio, sr, speaker_changes)

        # Create segments
        segments = self.create_segments(audio, sr, split_points)

        # Save segments
        segment_info = []
        speaker_dir = output_dir / speaker_id
        speaker_dir.mkdir(exist_ok=True)

        for start, end, time_range in segments:
            segment_audio = audio[start:end]

            # Save audio segment
            output_file = speaker_dir / f"{time_range}.wav"
            import soundfile as sf
            sf.write(output_file, segment_audio, sr)

            # Create placeholder lab file (to be filled with transcription)
            lab_file = speaker_dir / f"{time_range}.lab"
            lab_file.write_text("")

            info = {
                "file": str(output_file),
                "time_range": time_range,
                "duration": (end - start) / sr,
                "start_sample": start,
                "end_sample": end,
                "speaker": speaker_id,
            }
            segment_info.append(info)

        # Save metadata
        if save_metadata:
            metadata_file = output_dir / f"{speaker_id}_segments.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump({
                    "source_file": str(audio_path),
                    "speaker_id": speaker_id,
                    "num_segments": len(segment_info),
                    "total_duration": len(audio) / sr,
                    "segments": segment_info,
                }, f, indent=2, ensure_ascii=False)

        logger.info(f"Created {len(segment_info)} segments from {audio_path.name}")
        return segment_info

    def batch_segment(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        file_pattern: str = "*.wav",
        **kwargs,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Batch segment multiple audio files.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            file_pattern: Glob pattern for audio files
            **kwargs: Additional arguments for segment_long_audio

        Returns:
            Dictionary mapping filenames to segment info lists
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        for audio_file in sorted(input_dir.glob(file_pattern)):
            if not audio_file.is_file():
                continue

            speaker_id = audio_file.stem
            segments = self.segment_long_audio(
                audio_file, output_dir, speaker_id=speaker_id, **kwargs
            )
            results[audio_file.name] = segments

        return results


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Segment long audio files for voice cloning training"
    )
    parser.add_argument("input", help="Input audio file or directory")
    parser.add_argument("output", help="Output directory")
    parser.add_argument(
        "--min-duration", type=float, default=5.0,
        help="Minimum segment duration in seconds"
    )
    parser.add_argument(
        "--max-duration", type=float, default=15.0,
        help="Maximum segment duration in seconds"
    )
    parser.add_argument(
        "--target-duration", type=float, default=10.0,
        help="Target segment duration in seconds"
    )
    parser.add_argument(
        "--detect-speakers", action="store_true",
        help="Detect speaker change points"
    )
    parser.add_argument(
        "--pattern", default="*.wav",
        help="File pattern for batch processing"
    )

    args = parser.parse_args()

    segmenter = AudioSegmenter(
        min_segment_duration=args.min_duration,
        max_segment_duration=args.max_duration,
        target_segment_duration=args.target_duration,
    )

    input_path = Path(args.input)

    if input_path.is_file():
        segmenter.segment_long_audio(
            args.input,
            args.output,
            detect_speakers=args.detect_speakers,
        )
    else:
        segmenter.batch_segment(
            args.input,
            args.output,
            file_pattern=args.pattern,
            detect_speakers=args.detect_speakers,
        )
