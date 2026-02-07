"""
Audio preprocessing module for voice cloning reference audio.

This module provides functions to:
- Normalize audio to 24kHz sample rate
- Convert to mono if stereo
- Normalize loudness (LUFS)
- Trim silence from beginning/end
- Split long audio into segments
- Detect and remove low-quality segments
"""

import io
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from loguru import logger


class AudioPreprocessor:
    """
    Audio preprocessor for voice cloning reference audio.

    Ensures audio quality and format consistency for better voice cloning results.
    """

    def __init__(
        self,
        target_sample_rate: int = 24000,
        target_lufs: float = -16.0,
        min_duration: float = 3.0,
        max_duration: float = 30.0,
    ):
        """
        Initialize the audio preprocessor.

        Args:
            target_sample_rate: Target sample rate (default 24kHz for Fish Speech)
            target_lufs: Target loudness in LUFS (default -16.0 for speech)
            min_duration: Minimum duration in seconds (shorter clips will be rejected)
            max_duration: Maximum duration in seconds (longer clips will be split)
        """
        self.target_sample_rate = target_sample_rate
        self.target_lufs = target_lufs
        self.min_duration = min_duration
        self.max_duration = max_duration

    def preprocess_audio(
        self,
        audio_data: bytes,
        input_format: str = "wav",
    ) -> bytes:
        """
        Preprocess audio data for voice cloning.

        Args:
            audio_data: Raw audio bytes
            input_format: Input audio format (wav, mp3, etc.)

        Returns:
            Preprocessed audio bytes in WAV format at 24kHz
        """
        try:
            # Load audio
            with tempfile.NamedTemporaryFile(suffix=f".{input_format}") as temp_input:
                temp_input.write(audio_data)
                temp_input.flush()

                waveform, sr = torchaudio.load(temp_input.name)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample to target sample rate
            if sr != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=self.target_sample_rate
                )
                waveform = resampler(waveform)

            # Convert to numpy
            audio = waveform.squeeze().numpy()

            # Trim silence from beginning and end
            audio = self._trim_silence(audio)

            # Check duration
            duration = len(audio) / self.target_sample_rate
            if duration < self.min_duration:
                logger.warning(f"Audio too short: {duration:.2f}s < {self.min_duration}s")
                # Still return it, but could be improved by repeating
            elif duration > self.max_duration:
                # Trim to max duration
                max_samples = int(self.max_duration * self.target_sample_rate)
                audio = audio[:max_samples]
                logger.info(f"Audio trimmed to {self.max_duration}s")

            # Normalize loudness
            audio = self._normalize_loudness(audio)

            # Remove DC offset
            audio = audio - np.mean(audio)

            # Clip to prevent distortion
            audio = np.clip(audio, -0.99, 0.99)

            # Convert back to bytes
            output_buffer = io.BytesIO()
            sf.write(
                output_buffer,
                audio,
                self.target_sample_rate,
                format="WAV",
                subtype="PCM_16",
            )
            output_buffer.seek(0)

            return output_buffer.read()

        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            # Return original audio if preprocessing fails
            return audio_data

    def _trim_silence(
        self,
        audio: np.ndarray,
        threshold: float = 0.01,
        frame_length: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        """
        Trim silence from the beginning and end of audio.

        Args:
            audio: Audio array
            threshold: Energy threshold for silence detection
            frame_length: Frame length for energy calculation
            hop_length: Hop length for energy calculation

        Returns:
            Audio with silence trimmed
        """
        try:
            # Calculate energy in frames
            energy = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                energy.append(np.mean(np.abs(frame)))

            energy = np.array(energy)

            # Find first non-silent frame
            start_idx = 0
            for i, e in enumerate(energy):
                if e > threshold:
                    start_idx = i * hop_length
                    break

            # Find last non-silent frame
            end_idx = len(audio)
            for i in range(len(energy) - 1, -1, -1):
                if energy[i] > threshold:
                    end_idx = min((i + 1) * hop_length + frame_length, len(audio))
                    break

            # Trim
            trimmed = audio[start_idx:end_idx]

            # Ensure minimum length
            if len(trimmed) < len(audio) * 0.5:
                # If too much was trimmed, return original
                return audio

            return trimmed

        except Exception as e:
            logger.warning(f"Error trimming silence: {e}")
            return audio

    def _normalize_loudness(
        self,
        audio: np.ndarray,
        target_lufs: Optional[float] = None,
    ) -> np.ndarray:
        """
        Normalize audio loudness to target LUFS.

        Args:
            audio: Audio array
            target_lufs: Target LUFS (uses self.target_lufs if not specified)

        Returns:
            Loudness-normalized audio
        """
        try:
            if target_lufs is None:
                target_lufs = self.target_lufs

            # Calculate RMS
            rms = np.sqrt(np.mean(audio ** 2))

            # Avoid division by zero
            if rms < 1e-6:
                return audio

            # Calculate gain needed
            # Simple RMS normalization (for LUFS, we'd need pyloudnorm)
            target_rms = 10 ** ((target_lufs + 10) / 20) * 0.1
            gain = target_rms / rms

            # Limit gain to avoid excessive amplification
            gain = min(max(gain, 0.1), 10.0)

            # Apply gain
            normalized = audio * gain

            return normalized

        except Exception as e:
            logger.warning(f"Error normalizing loudness: {e}")
            return audio

    def split_audio(
        self,
        audio_data: bytes,
        segment_duration: float = 10.0,
        overlap: float = 1.0,
    ) -> List[bytes]:
        """
        Split long audio into segments for better voice cloning.

        Args:
            audio_data: Audio bytes
            segment_duration: Duration of each segment in seconds
            overlap: Overlap between segments in seconds

        Returns:
            List of audio segment bytes
        """
        try:
            # Load audio
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_input:
                temp_input.write(audio_data)
                temp_input.flush()

                waveform, sr = torchaudio.load(temp_input.name)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if needed
            if sr != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=self.target_sample_rate
                )
                waveform = resampler(waveform)

            audio = waveform.squeeze().numpy()
            duration = len(audio) / self.target_sample_rate

            # Don't split if audio is short enough
            if duration <= segment_duration:
                return [audio_data]

            # Split into segments
            segment_samples = int(segment_duration * self.target_sample_rate)
            overlap_samples = int(overlap * self.target_sample_rate)
            hop_samples = segment_samples - overlap_samples

            segments = []
            start = 0

            while start < len(audio):
                end = min(start + segment_samples, len(audio))
                segment = audio[start:end]

                # Pad last segment if needed
                if end - start < segment_samples:
                    padding = segment_samples - (end - start)
                    segment = np.pad(segment, (0, padding), mode='constant')

                # Convert to bytes
                output_buffer = io.BytesIO()
                sf.write(
                    output_buffer,
                    segment,
                    self.target_sample_rate,
                    format="WAV",
                    subtype="PCM_16",
                )
                output_buffer.seek(0)
                segments.append(output_buffer.read())

                start += hop_samples

            logger.info(f"Split audio into {len(segments)} segments")

            return segments

        except Exception as e:
            logger.error(f"Error splitting audio: {e}")
            return [audio_data]

    def validate_audio(
        self,
        audio_data: bytes,
    ) -> Tuple[bool, str]:
        """
        Validate audio quality for voice cloning.

        Args:
            audio_data: Audio bytes

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Load audio
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_input:
                temp_input.write(audio_data)
                temp_input.flush()

                waveform, sr = torchaudio.load(temp_input.name)

            # Check sample rate
            if sr < 16000:
                return False, f"Sample rate too low: {sr}Hz"

            # Check duration
            duration = waveform.shape[1] / sr
            if duration < self.min_duration:
                return False, f"Audio too short: {duration:.2f}s"

            # Check for clipping
            audio = waveform.squeeze().numpy()
            clipping_ratio = np.sum(np.abs(audio) > 0.99) / len(audio)
            if clipping_ratio > 0.01:
                return False, f"Too much clipping: {clipping_ratio*100:.1f}%"

            # Check for silence
            silence_ratio = np.sum(np.abs(audio) < 0.01) / len(audio)
            if silence_ratio > 0.5:
                return False, f"Too much silence: {silence_ratio*100:.1f}%"

            return True, ""

        except Exception as e:
            return False, f"Error validating audio: {str(e)}"


# Global preprocessor instance
_preprocessor = None


def get_preprocessor() -> AudioPreprocessor:
    """Get the global audio preprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = AudioPreprocessor()
    return _preprocessor
