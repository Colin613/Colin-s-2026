"""
Audio post-processing effects for TTS output enhancement.

This module provides audio effects for:
- Speed adjustment (time-stretching)
- Pitch shifting
- Volume normalization/gain
- Emotion intensity modulation
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
from typing import Optional

from loguru import logger


class AudioEffects:
    """
    Audio effects processor for TTS output.

    All effects maintain audio quality through proper signal processing.
    """

    def __init__(self, sample_rate: int = 24000):
        """
        Initialize audio effects processor.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate

    def adjust_speed(
        self,
        audio: np.ndarray,
        speed_factor: float = 1.0,
    ) -> np.ndarray:
        """
        Adjust audio speed using phase vocoder technique.

        This preserves pitch when changing speed, unlike naive resampling.

        Args:
            audio: Input audio array (float32, -1 to 1)
            speed_factor: Speed multiplier (0.5 = half speed, 2.0 = double speed)

        Returns:
            Speed-adjusted audio array
        """
        if speed_factor == 1.0:
            return audio

        logger.debug(f"Adjusting speed by factor: {speed_factor}")

        # Use scipy's resample for simple speed adjustment
        # For higher quality, consider using librosa.effects.time_stretch
        try:
            import librosa

            # Use librosa's time stretch (preserves pitch)
            stretched = librosa.effects.time_stretch(
                audio.astype(np.float32),
                rate=speed_factor,
            )
            return stretched

        except ImportError:
            # Fallback to simple resampling (changes pitch slightly)
            logger.warning("librosa not available, using simple resampling")

            original_length = len(audio)
            new_length = int(original_length / speed_factor)

            # Use scipy signal resample
            resampled = signal.resample(audio, new_length)

            return resampled

    def adjust_pitch(
        self,
        audio: np.ndarray,
        pitch_factor: float = 1.0,
    ) -> np.ndarray:
        """
        Adjust audio pitch using resampling with time correction.

        Args:
            audio: Input audio array (float32, -1 to 1)
            pitch_factor: Pitch multiplier (0.8 = lower, 1.2 = higher)

        Returns:
            Pitch-shifted audio array
        """
        if pitch_factor == 1.0:
            return audio

        logger.debug(f"Adjusting pitch by factor: {pitch_factor}")

        try:
            import librosa

            # Use librosa's pitch shift (preserves duration)
            shifted = librosa.effects.pitch_shift(
                audio.astype(np.float32),
                sr=self.sample_rate,
                n_steps=12 * np.log2(pitch_factor),  # Convert to semitones
            )
            return shifted

        except ImportError:
            # Fallback to simple resampling (changes duration)
            logger.warning("librosa not available, using simple resampling")

            # Resample to change pitch
            resampled = signal.resample(
                audio,
                int(len(audio) / pitch_factor),
            )

            # Time-stretch back to original length
            final = signal.resample(
                resampled,
                len(audio),
            )

            return final

    def adjust_volume(
        self,
        audio: np.ndarray,
        volume_gain: float = 1.0,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Adjust audio volume with optional normalization.

        Args:
            audio: Input audio array (float32, -1 to 1)
            volume_gain: Volume multiplier (0.5 = half, 2.0 = double)
            normalize: Whether to normalize peak to target level

        Returns:
            Volume-adjusted audio array
        """
        if volume_gain == 1.0 and not normalize:
            return audio

        logger.debug(f"Adjusting volume by gain: {volume_gain}")

        # Apply gain
        adjusted = audio * volume_gain

        # Normalize if requested
        if normalize:
            # Normalize to -1 dB
            peak = np.max(np.abs(adjusted))
            if peak > 0:
                target_peak = 0.89  # -1 dB
                adjusted = adjusted / peak * target_peak

        # Soft clip to prevent distortion
        adjusted = np.clip(adjusted, -1.0, 1.0)

        return adjusted

    def apply_emotion_intensity(
        self,
        audio: np.ndarray,
        emotion_intensity: float = 1.0,
    ) -> np.ndarray:
        """
        Apply emotion intensity modulation to audio.

        This enhances dynamic range for higher intensity.

        Args:
            audio: Input audio array (float32, -1 to 1)
            emotion_intensity: Intensity multiplier (0.5 = subdued, 1.5 = exaggerated)

        Returns:
            Emotion-modulated audio array
        """
        if emotion_intensity == 1.0:
            return audio

        logger.debug(f"Applying emotion intensity: {emotion_intensity}")

        # Calculate dynamic range adjustment
        if emotion_intensity > 1.0:
            # Increase dynamic range (more expressive)
            factor = emotion_intensity
            # Compress slightly then expand
            audio = np.tanh(audio * factor) / np.tanh(factor)
            # Add slight boost to peaks
            audio = audio * (1 + (emotion_intensity - 1.0) * 0.2)
        else:
            # Decrease dynamic range (more subdued)
            # Soft compression
            audio = np.tanh(audio / emotion_intensity) * emotion_intensity

        return np.clip(audio, -1.0, 1.0)

    def apply_all_effects(
        self,
        audio: np.ndarray,
        speed_factor: float = 1.0,
        pitch_factor: float = 1.0,
        emotion_intensity: float = 1.0,
        volume_gain: float = 1.0,
        normalize_output: bool = True,
    ) -> np.ndarray:
        """
        Apply all audio effects in sequence.

        Order of operations:
        1. Speed adjustment
        2. Pitch adjustment
        3. Emotion intensity modulation
        4. Volume adjustment

        Args:
            audio: Input audio array (float32, -1 to 1)
            speed_factor: Speed multiplier
            pitch_factor: Pitch multiplier
            emotion_intensity: Emotion intensity multiplier
            volume_gain: Volume gain multiplier
            normalize_output: Whether to normalize final output

        Returns:
            Processed audio array
        """
        result = audio.astype(np.float32)

        # Apply effects in sequence
        if speed_factor != 1.0:
            result = self.adjust_speed(result, speed_factor)

        if pitch_factor != 1.0:
            result = self.adjust_pitch(result, pitch_factor)

        if emotion_intensity != 1.0:
            result = self.apply_emotion_intensity(result, emotion_intensity)

        if volume_gain != 1.0:
            result = self.adjust_volume(result, volume_gain, normalize=normalize_output)

        # Final soft clipping
        result = np.clip(result, -1.0, 1.0)

        logger.debug(f"Applied effects: speed={speed_factor}, pitch={pitch_factor}, "
                    f"emotion={emotion_intensity}, volume={volume_gain}")

        return result

    def fade_in_out(
        self,
        audio: np.ndarray,
        fade_duration: float = 0.05,
    ) -> np.ndarray:
        """
        Apply fade in and fade out to prevent clicking.

        Args:
            audio: Input audio array
            fade_duration: Fade duration in seconds

        Returns:
            Audio with fades applied
        """
        fade_samples = int(fade_duration * self.sample_rate)
        fade_samples = min(fade_samples, len(audio) // 2)

        if fade_samples == 0:
            return audio

        # Create fade envelopes
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)

        result = audio.copy()

        # Apply fades
        result[:fade_samples] *= fade_in
        result[-fade_samples:] *= fade_out

        return result

    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC offset from audio."""
        return audio - np.mean(audio)

    def apply_highpass_filter(
        self,
        audio: np.ndarray,
        cutoff: float = 80.0,
    ) -> np.ndarray:
        """
        Apply high-pass filter to remove low-frequency rumble.

        Args:
            audio: Input audio array
            cutoff: Cutoff frequency in Hz

        Returns:
            Filtered audio
        """
        # Design Butterworth high-pass filter
        sos = signal.butter(4, cutoff, btype="high", fs=self.sample_rate, output="sos")
        filtered = signal.sosfilt(sos, audio)

        return filtered


def process_audio_with_effects(
    audio: np.ndarray,
    sample_rate: int,
    speed_factor: float = 1.0,
    pitch_factor: float = 1.0,
    emotion_intensity: float = 1.0,
    volume_gain: float = 1.0,
    apply_fades: bool = True,
) -> np.ndarray:
    """
    Convenience function to process audio with all effects.

    Args:
        audio: Input audio array
        sample_rate: Sample rate in Hz
        speed_factor: Speed multiplier
        pitch_factor: Pitch multiplier
        emotion_intensity: Emotion intensity multiplier
        volume_gain: Volume gain multiplier
        apply_fades: Whether to apply fade in/out

    Returns:
        Processed audio array
    """
    processor = AudioEffects(sample_rate)

    result = processor.apply_all_effects(
        audio,
        speed_factor=speed_factor,
        pitch_factor=pitch_factor,
        emotion_intensity=emotion_intensity,
        volume_gain=volume_gain,
        normalize_output=True,
    )

    # Apply fades for smooth output
    if apply_fades:
        result = processor.fade_in_out(result)

    # Remove DC offset
    result = processor.remove_dc_offset(result)

    # Clean up low frequencies
    result = processor.apply_highpass_filter(result)

    return result


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    import soundfile as sf

    parser = argparse.ArgumentParser(
        description="Apply audio effects to audio file"
    )
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Speed factor (0.5-2.0)"
    )
    parser.add_argument(
        "--pitch", type=float, default=1.0,
        help="Pitch factor (0.8-1.2)"
    )
    parser.add_argument(
        "--emotion", type=float, default=1.0,
        help="Emotion intensity (0.5-1.5)"
    )
    parser.add_argument(
        "--volume", type=float, default=1.0,
        help="Volume gain (0.5-2.0)"
    )

    args = parser.parse_args()

    # Load audio
    audio, sr = sf.read(args.input)

    # Process
    processed = process_audio_with_effects(
        audio,
        sr,
        speed_factor=args.speed,
        pitch_factor=args.pitch,
        emotion_intensity=args.emotion,
        volume_gain=args.volume,
    )

    # Save
    sf.write(args.output, processed, sr)
    print(f"Processed audio saved to: {args.output}")
