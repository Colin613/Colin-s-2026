"""
Audio post-processing module for TTS output enhancement.

This module provides audio effects for controlling the output audio:
- Speed adjustment
- Pitch shifting
- Volume gain
- Emotion intensity modulation
"""

from .audio_effects import (
    AudioEffects,
    process_audio_with_effects,
)

__all__ = [
    "AudioEffects",
    "process_audio_with_effects",
]
