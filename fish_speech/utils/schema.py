import base64
import os
import queue
from dataclasses import dataclass
from typing import Literal

import torch
from pydantic import BaseModel, Field, conint, model_validator
from pydantic.functional_validators import SkipValidation
from typing_extensions import Annotated

from fish_speech.content_sequence import TextPart, VQPart


class ServeVQPart(BaseModel):
    type: Literal["vq"] = "vq"
    codes: SkipValidation[list[list[int]]]


class ServeTextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ServeAudioPart(BaseModel):
    type: Literal["audio"] = "audio"
    audio: bytes


class ServeRequest(BaseModel):
    # Raw content sequence dict that we can use with ContentSequence(**content)
    content: dict
    max_new_tokens: int = 600
    top_p: float = 0.7
    repetition_penalty: float = 1.2
    temperature: float = 0.7
    streaming: bool = False
    num_samples: int = 1
    early_stop_threshold: float = 1.0


class ServeVQGANEncodeRequest(BaseModel):
    # The audio here should be in wav, mp3, etc
    audios: list[bytes]


class ServeVQGANEncodeResponse(BaseModel):
    tokens: SkipValidation[list[list[list[int]]]]


class ServeVQGANDecodeRequest(BaseModel):
    tokens: SkipValidation[list[list[list[int]]]]


class ServeVQGANDecodeResponse(BaseModel):
    # The audio here should be in PCM float16 format
    audios: list[bytes]


class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str

    @model_validator(mode="before")
    def decode_audio(cls, values):
        audio = values.get("audio")
        if (
            isinstance(audio, str) and len(audio) > 255
        ):  # Check if audio is a string (Base64)
            try:
                values["audio"] = base64.b64decode(audio)
            except Exception:
                # If the audio is not a valid base64 string, we will just ignore it and let the server handle it
                pass
        return values

    def __repr__(self) -> str:
        return f"ServeReferenceAudio(text={self.text!r}, audio_size={len(self.audio)})"


class ServeTTSRequest(BaseModel):
    text: str
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200
    # Audio format
    format: Literal["wav", "pcm", "mp3"] = "wav"
    # References audios for in-context learning
    references: list[ServeReferenceAudio] = []
    # Reference id
    # For example, if you want use https://fish.audio/m/7f92f8afb8ec43bf81429cc1c9199cb1/
    # Just pass 7f92f8afb8ec43bf81429cc1c9199cb1
    reference_id: str | None = None
    seed: int | None = None
    use_memory_cache: Literal["on", "off"] = "off"
    # Normalize text for en & zh, this increase stability for numbers
    normalize: bool = True
    # not usually used below
    streaming: bool = False
    max_new_tokens: int = 1024
    top_p: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.8
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2.0, strict=True)] = 1.1
    temperature: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.8

    # Audio post-processing parameters
    # Speed factor: 0.5 = half speed, 1.0 = normal, 2.0 = double speed
    speed_factor: Annotated[float, Field(ge=0.5, le=2.0, strict=True)] = 1.0
    # Pitch factor: 0.8 = lower pitch, 1.0 = normal, 1.2 = higher pitch
    pitch_factor: Annotated[float, Field(ge=0.8, le=1.2, strict=True)] = 1.0
    # Emotion intensity: 0.5 = subdued, 1.0 = normal, 1.5 = exaggerated
    emotion_intensity: Annotated[float, Field(ge=0.5, le=1.5, strict=True)] = 1.0
    # Volume gain: 0.5 = half volume, 1.0 = normal, 2.0 = double volume
    volume_gain: Annotated[float, Field(ge=0.5, le=2.0, strict=True)] = 1.0

    class Config:
        # Allow arbitrary types for pytorch related types
        arbitrary_types_allowed = True


class AddReferenceRequest(BaseModel):
    id: str = Field(..., min_length=1, max_length=255, pattern=r"^[a-zA-Z0-9\-_ ]+$")
    audio: bytes
    text: str = Field(..., min_length=1)


class AddReferenceResponse(BaseModel):
    success: bool
    message: str
    reference_id: str


class ListReferencesResponse(BaseModel):
    success: bool
    reference_ids: list[str]
    message: str = "Success"


class DeleteReferenceResponse(BaseModel):
    success: bool
    message: str
    reference_id: str


class UpdateReferenceResponse(BaseModel):
    success: bool
    message: str
    old_reference_id: str
    new_reference_id: str


# ==============================================================================
# Voice Library Schemas
# ==============================================================================

class VoiceInfo(BaseModel):
    id: str
    name: str
    description: str = ""
    language: str = "ko"  # Default to Korean
    created_at: str
    sample_rate: int = 24000
    duration: float = 0.0
    is_trained: bool = False
    audio_files: list = []  # List of audio file names
    reference_text: str = ""  # Reference text for voice cloning


class CreateVoiceRequest(BaseModel):
    id: str = Field(..., min_length=1, max_length=255, pattern=r"^[a-zA-Z0-9\-_ ]+$")
    name: str = Field(..., min_length=1, max_length=255)
    description: str = ""
    language: str = "ko"


class CreateVoiceResponse(BaseModel):
    success: bool
    message: str
    voice_id: str


class UpdateVoiceRequest(BaseModel):
    id: str = Field(..., min_length=1)
    name: str | None = None
    description: str | None = None
    language: str | None = None


class UpdateVoiceResponse(BaseModel):
    success: bool
    message: str
    voice_id: str


class ListVoicesResponse(BaseModel):
    success: bool
    voices: list[VoiceInfo]
    message: str = "Success"


class DeleteVoiceRequest(BaseModel):
    id: str = Field(..., min_length=1, description="Voice ID to delete")


class DeleteVoiceResponse(BaseModel):
    success: bool
    message: str
    voice_id: str


# ==============================================================================
# Training Task Schemas
# ==============================================================================

class TrainingTaskInfo(BaseModel):
    task_id: str
    voice_id: str
    voice_name: str | None = None
    status: str
    progress: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None
    checkpoint_path: str | None = None
    learning_rate: float | None = None
    batch_size: int | None = None


class CreateTrainingRequest(BaseModel):
    voice_id: str = Field(..., min_length=1)
    data_path: str = Field(..., min_length=1)
    max_steps: int = 5000
    batch_size: int = 16
    learning_rate: float = 1e-4


class CreateTrainingResponse(BaseModel):
    success: bool
    message: str
    task_id: str


class GetTrainingStatusResponse(BaseModel):
    success: bool
    task: TrainingTaskInfo | None = None
    message: str = "Success"


class CancelTrainingResponse(BaseModel):
    success: bool
    message: str
    task_id: str


class ListTrainingTasksResponse(BaseModel):
    success: bool
    tasks: list[TrainingTaskInfo]
    message: str = "Success"


# ==============================================================================
# Batch Dubbing Schemas
# ==============================================================================

class BatchJobInfo(BaseModel):
    job_id: str
    name: str
    status: str
    progress: float = 0.0
    total_items: int = 0
    completed_items: int = 0
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None
    output_path: str | None = None


class CreateBatchJobRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    subtitle_file: str = Field(..., min_length=1)  # SRT file path
    voice_mappings: dict[str, str] = Field(default_factory=dict)  # character -> voice_id
    output_format: str = "wav"


class CreateBatchJobResponse(BaseModel):
    success: bool
    message: str
    job_id: str


class GetBatchJobStatusResponse(BaseModel):
    success: bool
    job: BatchJobInfo | None = None
    message: str = "Success"


class ListBatchJobsResponse(BaseModel):
    success: bool
    jobs: list[BatchJobInfo]
    message: str = "Success"


class CancelBatchJobResponse(BaseModel):
    success: bool
    message: str
    job_id: str
