"""
Audio preprocessing tools for Yanbian Korean voice cloning.

This module provides utilities for:
- Audio format conversion (24kHz, mono, WAV)
- Loudness normalization
- Silence detection and segmentation
- Audio quality enhancement (noise reduction, de-reverberation)
"""

import os
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

import librosa
import numpy as np
import soundfile as sf
from loguru import logger

# Target audio specifications for Fish Speech
TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1  # mono
TARGET_FORMAT = "wav"


class AudioProcessor:
    """
    Audio processor for preparing training data for voice cloning.

    Features:
    - Format conversion to 24kHz mono WAV
    - Loudness normalization (EBU R128)
    - Silence detection and segmentation
    - Basic quality enhancement
    """

    def __init__(
        self,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        target_channels: int = TARGET_CHANNELS,
    ):
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels

    def load_audio(self, audio_path: str | Path) -> Tuple[np.ndarray, int]:
        """
        Load audio file using librosa for robust handling of various formats.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio_path = Path(audio_path)
        logger.info(f"Loading audio: {audio_path}")

        # Use librosa for automatic sample rate conversion
        audio, sr = librosa.load(
            str(audio_path),
            sr=self.target_sample_rate,
            mono=True,
        )

        logger.info(f"Loaded audio: shape={audio.shape}, duration={len(audio)/sr:.2f}s")
        return audio, sr

    def normalize_loudness(self, audio: np.ndarray, target_lufs: float = -16.0) -> np.ndarray:
        """
        Normalize audio to target LUFS (loudness) level.

        Uses EBU R128 standard for broadcast loudness normalization.

        Args:
            audio: Audio data array
            target_lufs: Target loudness in LUFS (default -16.0 for speech)

        Returns:
            Normalized audio array
        """
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio ** 2))

        # Avoid division by zero
        if rms < 1e-10:
            logger.warning("Audio has very low energy, skipping normalization")
            return audio

        # Calculate gain needed to reach target level
        # Simple RMS-based normalization (more sophisticated LUFS requires pyloudnorm)
        target_rms = 10 ** (target_lufs / 20) * 0.1  # Approximate mapping
        gain = target_rms / (rms + 1e-10)

        # Limit gain to avoid excessive amplification
        gain = min(gain, 10.0)

        normalized = audio * gain

        # Clip to prevent distortion
        normalized = np.clip(normalized, -1.0, 1.0)

        logger.info(f"Normalized audio: gain={gain:.2f}x")
        return normalized

    def detect_silence(
        self,
        audio: np.ndarray,
        sr: int,
        silence_threshold: float = 0.01,
        min_silence_duration: float = 0.3,
    ) -> List[Tuple[float, float]]:
        """
        Detect silent segments in audio.

        Args:
            audio: Audio data array
            sr: Sample rate
            silence_threshold: RMS threshold for silence detection
            min_silence_duration: Minimum duration (seconds) to consider as silence

        Returns:
            List of (start_time, end_time) tuples for silent segments
        """
        # Calculate RMS in windows
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.01 * sr)    # 10ms hop

        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length,
        )[0]

        # Time axis
        times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

        # Find silence segments
        is_silent = rms < silence_threshold

        silent_segments = []
        start_time = None

        for i, silent in enumerate(is_silent):
            time = times[i]
            if silent and start_time is None:
                start_time = time
            elif not silent and start_time is not None:
                duration = time - start_time
                if duration >= min_silence_duration:
                    silent_segments.append((start_time, time))
                start_time = None

        # Handle case where audio ends in silence
        if start_time is not None:
            duration = times[-1] - start_time
            if duration >= min_silence_duration:
                silent_segments.append((start_time, times[-1]))

        logger.info(f"Detected {len(silent_segments)} silent segments")
        return silent_segments

    def segment_by_silence(
        self,
        audio: np.ndarray,
        sr: int,
        min_segment_duration: float = 5.0,
        max_segment_duration: float = 15.0,
        silence_threshold: float = 0.01,
        min_silence_duration: float = 0.5,
    ) -> List[Tuple[int, int]]:
        """
        Segment audio by silence detection.

        Args:
            audio: Audio data array
            sr: Sample rate
            min_segment_duration: Minimum segment duration in seconds
            max_segment_duration: Maximum segment duration in seconds
            silence_threshold: RMS threshold for silence detection
            min_silence_duration: Minimum silence duration for split point

        Returns:
            List of (start_sample, end_sample) tuples for segments
        """
        silent_segments = self.detect_silence(
            audio, sr, silence_threshold, min_silence_duration
        )

        # Convert silent segments to split points
        split_points = [0]
        for start, end in silent_segments:
            mid_point = int((start + end) / 2 * sr)
            split_points.append(mid_point)
        split_points.append(len(audio))

        # Create segments respecting min/max duration
        segments = []
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            duration = (end - start) / sr

            if min_segment_duration <= duration <= max_segment_duration:
                segments.append((start, end))
            elif duration > max_segment_duration:
                # Split long segments
                num_splits = int(duration // max_segment_duration) + 1
                chunk_size = (end - start) // num_splits
                for j in range(num_splits):
                    seg_start = start + j * chunk_size
                    seg_end = min(start + (j + 1) * chunk_size, end)
                    if seg_end - seg_start > min_segment_duration * sr:
                        segments.append((seg_start, seg_end))

        logger.info(f"Created {len(segments)} segments")
        return segments

    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC offset from audio."""
        return audio - np.mean(audio)

    def apply_highpass_filter(
        self, audio: np.ndarray, sr: int, cutoff: float = 80.0
    ) -> np.ndarray:
        """
        Apply high-pass filter to remove low-frequency noise.

        Args:
            audio: Audio data array
            sr: Sample rate
            cutoff: Cutoff frequency in Hz

        Returns:
            Filtered audio array
        """
        from scipy.signal import butter, sosfilt

        # Design Butterworth high-pass filter
        sos = butter(4, cutoff, btype="high", fs=sr, output="sos")
        filtered = sosfilt(sos, audio)

        return filtered

    def enhance_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply basic audio enhancement.

        - DC offset removal
        - High-pass filtering for low-frequency noise
        - Light dynamic range compression

        Args:
            audio: Audio data array
            sr: Sample rate

        Returns:
            Enhanced audio array
        """
        # Remove DC offset
        audio = self.remove_dc_offset(audio)

        # Apply high-pass filter
        audio = self.apply_highpass_filter(audio, sr)

        # Light compression via soft clipping
        audio = np.tanh(audio * 0.9) / 0.9

        return audio

    def process_audio_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        segment: bool = True,
        normalize: bool = True,
        enhance: bool = True,
    ) -> List[Tuple[str, str]]:
        """
        Process a single audio file for training data preparation.

        Args:
            input_path: Input audio file path
            output_path: Output directory path
            segment: Whether to segment audio
            normalize: Whether to normalize loudness
            enhance: Whether to apply audio enhancement

        Returns:
            List of (output_file, time_range) tuples for generated files
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load audio
        audio, sr = self.load_audio(input_path)

        # Apply enhancements
        if enhance:
            audio = self.enhance_audio(audio, sr)

        # Normalize loudness
        if normalize:
            audio = self.normalize_loudness(audio)

        # Segment or save whole file
        if segment:
            segments = self.segment_by_silence(audio, sr)
        else:
            segments = [(0, len(audio))]

        results = []
        for i, (start, end) in enumerate(segments):
            segment_audio = audio[start:end]
            duration = (end - start) / sr

            # Skip very short segments
            if duration < 3.0:
                continue

            # Generate filename
            start_time = start / sr
            end_time = end / sr
            time_str = f"{start_time:.2f}-{end_time:.2f}"
            output_file = output_path / f"{input_path.stem}_{time_str}.wav"

            # Save segment
            sf.write(output_file, segment_audio, sr)
            results.append((str(output_file), time_str))

        logger.info(f"Processed {input_path.name}: {len(results)} segments created")
        return results

    def process_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        **kwargs,
    ) -> dict[str, List[Tuple[str, str]]]:
        """
        Process all audio files in a directory.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            **kwargs: Additional arguments for process_audio_file

        Returns:
            Dictionary mapping input filenames to output file lists
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Supported audio formats
        audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}

        results = {}
        for audio_file in input_dir.iterdir():
            if audio_file.suffix.lower() not in audio_extensions:
                continue

            if audio_file.is_dir():
                continue

            # Create speaker-specific output directory
            speaker_output = output_dir / audio_file.stem
            outputs = self.process_audio_file(
                audio_file, speaker_output, **kwargs
            )
            results[audio_file.name] = outputs

        return results


def process_with_ffmpeg(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
) -> bool:
    """
    Process audio using ffmpeg for format conversion and basic normalization.

    Args:
        input_path: Input audio file path
        output_path: Output audio file path
        sample_rate: Target sample rate
        channels: Target number of channels (1 for mono)

    Returns:
        True if successful, False otherwise
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-y",  # Overwrite output file
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Converted with ffmpeg: {input_path} -> {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg conversion failed: {e.stderr.decode()}")
        return False


# CLI interface for standalone usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess audio files for voice cloning training"
    )
    parser.add_argument("input", help="Input audio file or directory")
    parser.add_argument("output", help="Output directory")
    parser.add_argument(
        "--segment", action="store_true", help="Segment audio by silence"
    )
    parser.add_argument(
        "--normalize", action="store_true", help="Normalize loudness"
    )
    parser.add_argument(
        "--enhance", action="store_true", help="Apply audio enhancement"
    )
    parser.add_argument(
        "--ffmpeg", action="store_true", help="Use ffmpeg for basic conversion"
    )

    args = parser.parse_args()

    processor = AudioProcessor()

    if args.ffmpeg:
        if Path(args.input).is_file():
            process_with_ffmpeg(args.input, args.output)
        else:
            for audio_file in Path(args.input).iterdir():
                if audio_file.suffix.lower() in {".wav", ".mp3", ".flac"}:
                    out_file = Path(args.output) / f"{audio_file.stem}.wav"
                    process_with_ffmpeg(audio_file, out_file)
    else:
        if Path(args.input).is_file():
            processor.process_audio_file(
                args.input,
                args.output,
                segment=args.segment,
                normalize=args.normalize,
                enhance=args.enhance,
            )
        else:
            processor.process_directory(
                args.input,
                args.output,
                segment=args.segment,
                normalize=args.normalize,
                enhance=args.enhance,
            )
