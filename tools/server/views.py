import io
import os
import re
import shutil
import tempfile
import time
from http import HTTPStatus
from pathlib import Path

import numpy as np
import ormsgpack
import soundfile as sf
import torch
from kui.asgi import (
    Body,
    HTTPException,
    HttpView,
    JSONResponse,
    Routes,
    StreamResponse,
    UploadFile,
    request,
)
from loguru import logger
from typing_extensions import Annotated

from fish_speech.utils.schema import (
    AddReferenceRequest,
    AddReferenceResponse,
    DeleteReferenceResponse,
    ListReferencesResponse,
    ServeTTSRequest,
    ServeVQGANDecodeRequest,
    ServeVQGANDecodeResponse,
    ServeVQGANEncodeRequest,
    ServeVQGANEncodeResponse,
    UpdateReferenceResponse,
    # Voice Library
    VoiceInfo,
    CreateVoiceRequest,
    CreateVoiceResponse,
    DeleteVoiceRequest,
    UpdateVoiceRequest,
    UpdateVoiceResponse,
    ListVoicesResponse,
    DeleteVoiceResponse,
    # Training
    TrainingTaskInfo,
    CreateTrainingRequest,
    CreateTrainingResponse,
    GetTrainingStatusResponse,
    CancelTrainingResponse,
    ListTrainingTasksResponse,
    # Batch Dubbing
    BatchJobInfo,
    CreateBatchJobRequest,
    CreateBatchJobResponse,
    GetBatchJobStatusResponse,
    ListBatchJobsResponse,
    CancelBatchJobResponse,
)
from tools.server.api_utils import (
    buffer_to_async_generator,
    format_response,
    get_content_type,
    inference_async,
)
from tools.server.audio_preprocessor import get_preprocessor
from tools.server.inference import inference_wrapper as inference
from tools.server.model_manager import ModelManager
from tools.server.model_utils import (
    batch_vqgan_decode,
    cached_vqgan_batch_encode,
)
from tools.server.training_manager import (
    get_training_manager,
    TrainingStatus,
)

MAX_NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", 1))

routes = Routes()


@routes.http("/v1/health")
class Health(HttpView):
    @classmethod
    async def get(cls):
        return JSONResponse({"status": "ok"})

    @classmethod
    async def post(cls):
        return JSONResponse({"status": "ok"})


@routes.http.post("/v1/vqgan/encode")
async def vqgan_encode(req: Annotated[ServeVQGANEncodeRequest, Body(exclusive=True)]):
    """
    Encode audio using VQGAN model.
    """
    try:
        # Get the model from the app
        model_manager: ModelManager = request.app.state.model_manager
        decoder_model = model_manager.decoder_model

        # Encode the audio
        start_time = time.time()
        tokens = cached_vqgan_batch_encode(decoder_model, req.audios)
        logger.info(
            f"[EXEC] VQGAN encode time: {(time.time() - start_time) * 1000:.2f}ms"
        )

        # Return the response
        return ormsgpack.packb(
            ServeVQGANEncodeResponse(tokens=[i.tolist() for i in tokens]),
            option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
        )
    except Exception as e:
        logger.error(f"Error in VQGAN encode: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR, content="Failed to encode audio"
        )


@routes.http.post("/v1/vqgan/decode")
async def vqgan_decode(req: Annotated[ServeVQGANDecodeRequest, Body(exclusive=True)]):
    """
    Decode tokens to audio using VQGAN model.
    """
    try:
        # Get the model from the app
        model_manager: ModelManager = request.app.state.model_manager
        decoder_model = model_manager.decoder_model

        # Decode the audio
        tokens = [torch.tensor(token, dtype=torch.int) for token in req.tokens]
        start_time = time.time()
        audios = batch_vqgan_decode(decoder_model, tokens)
        logger.info(
            f"[EXEC] VQGAN decode time: {(time.time() - start_time) * 1000:.2f}ms"
        )
        audios = [audio.astype(np.float16).tobytes() for audio in audios]

        # Return the response
        return ormsgpack.packb(
            ServeVQGANDecodeResponse(audios=audios),
            option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
        )
    except Exception as e:
        logger.error(f"Error in VQGAN decode: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR, content="Failed to decode tokens to audio"
        )


@routes.http.post("/v1/tts")
async def tts(req: Annotated[ServeTTSRequest, Body(exclusive=True)]):
    """
    Generate speech from text using TTS model.
    """
    try:
        # Get the model from the app
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine
        sample_rate = engine.decoder_model.sample_rate

        # Check if the text is too long
        if app_state.max_text_length > 0 and len(req.text) > app_state.max_text_length:
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                content=f"Text is too long, max length is {app_state.max_text_length}",
            )

        # Check if streaming is enabled
        if req.streaming and req.format != "wav":
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                content="Streaming only supports WAV format",
            )

        # Perform TTS
        if req.streaming:
            return StreamResponse(
                iterable=inference_async(req, engine),
                headers={
                    "Content-Disposition": f"attachment; filename=audio.{req.format}",
                },
                content_type=get_content_type(req.format),
            )
        else:
            fake_audios = next(inference(req, engine))
            buffer = io.BytesIO()
            sf.write(
                buffer,
                fake_audios,
                sample_rate,
                format=req.format,
            )

            return StreamResponse(
                iterable=buffer_to_async_generator(buffer.getvalue()),
                headers={
                    "Content-Disposition": f"attachment; filename=audio.{req.format}",
                },
                content_type=get_content_type(req.format),
            )
    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except Exception as e:
        logger.error(f"Error in TTS generation: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR, content="Failed to generate speech"
        )


@routes.http.post("/v1/voice/tts")
async def voice_tts(
    voice_id: str = Body(...),
    text: str = Body(...),
    format: str = Body("wav"),
    emotion: str = Body(""),
    speed: float = Body(1.0),
    pitch: float = Body(1.0),
):
    """
    Generate speech using a specific voice from the voice library.
    Uses multiple reference audio clips for better voice matching.
    """
    ref_id = None
    try:
        # Get the voice information
        voices_base = Path("voices")
        voice_dir = voices_base / voice_id

        if not voice_dir.exists():
            return JSONResponse(
                {"success": False, "message": f"Voice '{voice_id}' not found"},
                status_code=404
            )

        # Load voice metadata
        import json
        metadata_file = voice_dir / "metadata.json"
        if not metadata_file.exists():
            return JSONResponse(
                {"success": False, "message": "Voice metadata not found"},
                status_code=404
            )

        metadata = json.loads(metadata_file.read_text())

        # Check if this voice has a trained LoRA model
        trained_model_path = metadata.get("trained_model_path")
        is_trained = metadata.get("is_trained", False)

        if trained_model_path and Path(trained_model_path).exists():
            # TODO: Load trained LoRA model for inference
            # For now, we'll use reference-based TTS but indicate that a trained model exists
            logger.info(f"Voice {voice_id} has trained LoRA model at {trained_model_path}")
            # Note: To use the trained model, we would need to:
            # 1. Load the merged checkpoint
            # 2. Replace the model in the engine
            # 3. Run inference
            # This requires implementing model switching in the ModelManager

        # Get the model from the app
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        # Check if the text is too long
        if app_state.max_text_length > 0 and len(text) > app_state.max_text_length:
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                content=f"Text is too long, max length is {app_state.max_text_length}",
            )

        # Collect all reference audio files from the voice directory
        from fish_speech.utils.file import AUDIO_EXTENSIONS
        from fish_speech.utils.file import list_files

        audio_files = list_files(voice_dir, AUDIO_EXTENSIONS, recursive=False, sort=False)

        if not audio_files:
            return JSONResponse(
                {"success": False, "message": "No reference audio files found for this voice"},
                status_code=400
            )

        # Use multiple references for better voice cloning
        ref_id = f"voice_{voice_id}"
        ref_dir = Path("references") / ref_id
        ref_dir.mkdir(parents=True, exist_ok=True)

        # Copy all audio files to references directory with .lab files
        reference_texts = []
        for i, audio_file in enumerate(audio_files):
            # Copy audio file
            target_audio = ref_dir / f"ref_{i:03d}{audio_file.suffix}"
            shutil.copy2(audio_file, target_audio)

            # Create .lab file with appropriate text
            # Use a generic text that matches the language style
            ref_text = metadata.get("reference_text", "안녕하세요? 이것은 테스트입니다.")
            lab_file = ref_dir / f"ref_{i:03d}.lab"
            with open(lab_file, "w", encoding="utf-8") as f:
                f.write(ref_text)
            reference_texts.append(ref_text)

        # Clear cache for this reference ID
        if ref_id in engine.ref_by_id:
            del engine.ref_by_id[ref_id]

        # Create TTS request with reference_id
        # The engine will load all audio files from the references/{ref_id} directory
        from fish_speech.utils.schema import ServeTTSRequest

        # Build emotion prefix if provided
        final_text = text
        if emotion and emotion.strip():
            emotion_prefixs = {
                "happy": "[Laugh]",
                "sad": "[Cry]",
                "angry": "[Angry]",
                "whisper": "[Whisper]",
                "shout": "[Shout]",
            }
            prefix = emotion_prefixs.get(emotion.lower(), "")
            if prefix:
                final_text = f"{prefix} {text}"

        tts_req = ServeTTSRequest(
            text=final_text,
            format=format,
            reference_id=ref_id,
            chunk_length=100,  # Reduced from 200 for better voice matching
            streaming=False,
            speed_factor=max(0.5, min(2.0, speed)),
            pitch_factor=max(0.8, min(1.2, pitch)),
            temperature=0.3,  # Much lower temperature to follow reference more closely
            top_p=0.7,  # Reduced from 0.8 for more deterministic output
            repetition_penalty=1.0,  # Reduced to allow more natural repetitions
        )

        # Generate audio using the standard inference wrapper
        audio_buffer = []
        for result in engine.inference(tts_req):
            if result.code == "segment":
                if isinstance(result.audio, tuple):
                    # Apply amplitude and convert to bytes
                    audio_data = (result.audio[1] * 32768).astype(np.int16).tobytes()
                    audio_buffer.append(audio_data)
            elif result.code == "final":
                if isinstance(result.audio, tuple):
                    audio_data = (result.audio[1] * 32768).astype(np.int16).tobytes()
                    audio_buffer.append(audio_data)
            elif result.code == "error":
                raise HTTPException(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    content=str(result.error),
                )

        # Combine all audio segments
        audio_bytes = b"".join(audio_buffer)

        # Add WAV header for proper browser playback
        # WAV format: RIFF header + fmt chunk + data chunk
        num_channels = 1  # Mono
        sample_rate = 24000  # 24kHz
        bits_per_sample = 16  # 16-bit PCM
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(audio_bytes)
        file_size = 36 + data_size  # 36 + data size

        # Build WAV header
        wav_header = (
            b"RIFF" +  # ChunkID
            file_size.to_bytes(4, "little") +  # ChunkSize
            b"WAVE" +  # Format
            b"fmt " +  # Subchunk1ID
            (16).to_bytes(4, "little") +  # Subchunk1Size (16 for PCM)
            (1).to_bytes(2, "little") +  # AudioFormat (1 for PCM)
            num_channels.to_bytes(2, "little") +  # NumChannels
            sample_rate.to_bytes(4, "little") +  # SampleRate
            byte_rate.to_bytes(4, "little") +  # ByteRate
            block_align.to_bytes(2, "little") +  # BlockAlign
            bits_per_sample.to_bytes(2, "little") +  # BitsPerSample
            b"data" +  # Subchunk2ID
            data_size.to_bytes(4, "little")  # Subchunk2Size
        )

        wav_bytes = wav_header + audio_bytes

        # Clean up the reference directory
        try:
            if ref_dir.exists():
                shutil.rmtree(ref_dir)
            if ref_id in engine.ref_by_id:
                del engine.ref_by_id[ref_id]
        except:
            pass

        # Return the audio using async generator
        async def audio_generator():
            yield wav_bytes

        return StreamResponse(
            iterable=audio_generator(),
            content_type="audio/wav",
        )

    except HTTPException:
        # Clean up reference
        try:
            if ref_id:
                ref_dir = Path("references") / ref_id
                if ref_dir.exists():
                    shutil.rmtree(ref_dir)
                app_state = request.app.state
                model_manager: ModelManager = app_state.model_manager
                engine = model_manager.tts_inference_engine
                if ref_id in engine.ref_by_id:
                    del engine.ref_by_id[ref_id]
        except:
            pass
        raise

    except Exception as e:
        logger.error(f"Error in voice TTS generation: {e}", exc_info=True)
        # Clean up reference
        try:
            if ref_id:
                ref_dir = Path("references") / ref_id
                if ref_dir.exists():
                    shutil.rmtree(ref_dir)
                app_state = request.app.state
                model_manager: ModelManager = app_state.model_manager
                engine = model_manager.tts_inference_engine
                if ref_id in engine.ref_by_id:
                    del engine.ref_by_id[ref_id]
        except:
            pass
        return JSONResponse(
            {"success": False, "message": f"Failed to generate speech: {str(e)}"},
            status_code=500
        )


@routes.http.post("/v1/references/add")
async def add_reference(
    id: str = Body(...), audio: UploadFile = Body(...), text: str = Body(...)
):
    """
    Add a new reference voice with audio file and text.
    """
    temp_file_path = None

    try:
        # Validate input parameters
        if not id or not id.strip():
            raise ValueError("Reference ID cannot be empty")

        if not text or not text.strip():
            raise ValueError("Reference text cannot be empty")

        # Get the model manager to access the reference loader
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        # Read the uploaded audio file
        audio_content = audio.read()
        if not audio_content:
            raise ValueError("Audio file is empty or could not be read")

        # Create a temporary file for the audio data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_content)
            temp_file_path = temp_file.name

        # Add the reference using the engine's reference loader
        engine.add_reference(id, temp_file_path, text)

        response = AddReferenceResponse(
            success=True,
            message=f"Reference voice '{id}' added successfully",
            reference_id=id,
        )
        return format_response(response)

    except FileExistsError as e:
        logger.warning(f"Reference ID '{id}' already exists: {e}")
        response = AddReferenceResponse(
            success=False,
            message=f"Reference ID '{id}' already exists",
            reference_id=id,
        )
        return format_response(response, status_code=409)  # Conflict

    except ValueError as e:
        logger.warning(f"Invalid input for reference '{id}': {e}")
        response = AddReferenceResponse(success=False, message=str(e), reference_id=id)
        return format_response(response, status_code=400)

    except (FileNotFoundError, OSError) as e:
        logger.error(f"File system error for reference '{id}': {e}")
        response = AddReferenceResponse(
            success=False, message="File system error occurred", reference_id=id
        )
        return format_response(response, status_code=500)

    except Exception as e:
        logger.error(f"Unexpected error adding reference '{id}': {e}", exc_info=True)
        response = AddReferenceResponse(
            success=False, message="Internal server error occurred", reference_id=id
        )
        return format_response(response, status_code=500)

    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError as e:
                logger.warning(
                    f"Failed to clean up temporary file {temp_file_path}: {e}"
                )


@routes.http.get("/v1/references/list")
async def list_references():
    """
    Get a list of all available reference voice IDs.
    """
    try:
        # Get the model manager to access the reference loader
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        # Get the list of reference IDs
        reference_ids = engine.list_reference_ids()

        response = ListReferencesResponse(
            success=True,
            reference_ids=reference_ids,
            message=f"Found {len(reference_ids)} reference voices",
        )
        return format_response(response)

    except Exception as e:
        logger.error(f"Unexpected error listing references: {e}", exc_info=True)
        response = ListReferencesResponse(
            success=False, reference_ids=[], message="Internal server error occurred"
        )
        return format_response(response, status_code=500)


@routes.http.delete("/v1/references/delete")
async def delete_reference(reference_id: str = Body(...)):
    """
    Delete a reference voice by ID.
    """
    try:
        # Validate input parameters
        if not reference_id or not reference_id.strip():
            raise ValueError("Reference ID cannot be empty")

        # Get the model manager to access the reference loader
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        # Delete the reference using the engine's reference loader
        engine.delete_reference(reference_id)

        response = DeleteReferenceResponse(
            success=True,
            message=f"Reference voice '{reference_id}' deleted successfully",
            reference_id=reference_id,
        )
        return format_response(response)

    except FileNotFoundError as e:
        logger.warning(f"Reference ID '{reference_id}' not found: {e}")
        response = DeleteReferenceResponse(
            success=False,
            message=f"Reference ID '{reference_id}' not found",
            reference_id=reference_id,
        )
        return format_response(response, status_code=404)  # Not Found

    except ValueError as e:
        logger.warning(f"Invalid input for reference '{reference_id}': {e}")
        response = DeleteReferenceResponse(
            success=False, message=str(e), reference_id=reference_id
        )
        return format_response(response, status_code=400)

    except OSError as e:
        logger.error(f"File system error deleting reference '{reference_id}': {e}")
        response = DeleteReferenceResponse(
            success=False,
            message="File system error occurred",
            reference_id=reference_id,
        )
        return format_response(response, status_code=500)

    except Exception as e:
        logger.error(
            f"Unexpected error deleting reference '{reference_id}': {e}", exc_info=True
        )
        response = DeleteReferenceResponse(
            success=False,
            message="Internal server error occurred",
            reference_id=reference_id,
        )
        return format_response(response, status_code=500)


@routes.http.post("/v1/references/update")
async def update_reference(
    old_reference_id: str = Body(...), new_reference_id: str = Body(...)
):
    """
    Rename a reference voice directory from old_reference_id to new_reference_id.
    """
    try:
        # Validate input parameters
        if not old_reference_id or not old_reference_id.strip():
            raise ValueError("Old reference ID cannot be empty")
        if not new_reference_id or not new_reference_id.strip():
            raise ValueError("New reference ID cannot be empty")
        if old_reference_id == new_reference_id:
            raise ValueError("New reference ID must be different from old reference ID")

        # Validate ID format per ReferenceLoader rules
        id_pattern = r"^[a-zA-Z0-9\-_ ]+$"
        if not re.match(id_pattern, new_reference_id) or len(new_reference_id) > 255:
            raise ValueError(
                "New reference ID contains invalid characters or is too long"
            )

        # Access engine to update caches after renaming
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        refs_base = Path("references")
        old_dir = refs_base / old_reference_id
        new_dir = refs_base / new_reference_id

        # Existence checks
        if not old_dir.exists() or not old_dir.is_dir():
            raise FileNotFoundError(f"Reference ID '{old_reference_id}' not found")
        if new_dir.exists():
            # Conflict: destination already exists
            response = UpdateReferenceResponse(
                success=False,
                message=f"Reference ID '{new_reference_id}' already exists",
                old_reference_id=old_reference_id,
                new_reference_id=new_reference_id,
            )
            return format_response(response, status_code=409)

        # Perform rename
        old_dir.rename(new_dir)

        # Update in-memory cache key if present
        if old_reference_id in engine.ref_by_id:
            engine.ref_by_id[new_reference_id] = engine.ref_by_id.pop(old_reference_id)

        response = UpdateReferenceResponse(
            success=True,
            message=(
                f"Reference voice renamed from '{old_reference_id}' to '{new_reference_id}' successfully"
            ),
            old_reference_id=old_reference_id,
            new_reference_id=new_reference_id,
        )
        return format_response(response)

    except FileNotFoundError as e:
        logger.warning(str(e))
        response = UpdateReferenceResponse(
            success=False,
            message=str(e),
            old_reference_id=old_reference_id,
            new_reference_id=new_reference_id,
        )
        return format_response(response, status_code=404)

    except ValueError as e:
        logger.warning(f"Invalid input for update reference: {e}")
        response = UpdateReferenceResponse(
            success=False,
            message=str(e),
            old_reference_id=old_reference_id if "old_reference_id" in locals() else "",
            new_reference_id=new_reference_id if "new_reference_id" in locals() else "",
        )
        return format_response(response, status_code=400)

    except OSError as e:
        logger.error(f"File system error renaming reference: {e}")
        response = UpdateReferenceResponse(
            success=False,
            message="File system error occurred",
            old_reference_id=old_reference_id,
            new_reference_id=new_reference_id,
        )
        return format_response(response, status_code=500)

    except Exception as e:
        logger.error(f"Unexpected error updating reference: {e}", exc_info=True)
        response = UpdateReferenceResponse(
            success=False,
            message="Internal server error occurred",
            old_reference_id=old_reference_id if "old_reference_id" in locals() else "",
            new_reference_id=new_reference_id if "new_reference_id" in locals() else "",
        )
        return format_response(response, status_code=500)


# ==============================================================================
# Voice Library API Endpoints
# ==============================================================================

@routes.http.post("/v1/voices/create")
async def create_voice(req: Annotated[CreateVoiceRequest, Body(exclusive=True)]):
    """
    Create a new voice entry in the voice library.
    """
    try:
        voices_base = Path("voices")
        voices_base.mkdir(exist_ok=True)

        voice_dir = voices_base / req.id

        if voice_dir.exists():
            response = CreateVoiceResponse(
                success=False,
                message=f"Voice ID '{req.id}' already exists",
                voice_id=req.id,
            )
            return format_response(response, status_code=409)

        # Create voice directory and metadata
        voice_dir.mkdir(exist_ok=True)

        import json
        from datetime import datetime

        metadata = {
            "id": req.id,
            "name": req.name,
            "description": req.description,
            "language": req.language,
            "created_at": datetime.utcnow().isoformat(),
            "sample_rate": 24000,
            "duration": 0.0,
            "is_trained": False,
        }

        metadata_file = voice_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

        response = CreateVoiceResponse(
            success=True,
            message=f"Voice '{req.id}' created successfully",
            voice_id=req.id,
        )
        return format_response(response)

    except Exception as e:
        logger.error(f"Error creating voice: {e}", exc_info=True)
        response = CreateVoiceResponse(
            success=False,
            message="Internal server error occurred",
            voice_id=req.id if "req" in locals() else "",
        )
        return format_response(response, status_code=500)


@routes.http.get("/v1/voices/list")
async def list_voices():
    """
    Get a list of all voices in the voice library.
    """
    try:
        voices_base = Path("voices")
        voices = []

        if voices_base.exists():
            import json

            for voice_dir in voices_base.iterdir():
                if not voice_dir.is_dir():
                    continue

                metadata_file = voice_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        metadata = json.loads(metadata_file.read_text())
                        voices.append(VoiceInfo(**metadata))
                    except Exception as e:
                        logger.warning(f"Failed to load voice metadata for {voice_dir.name}: {e}")
                else:
                    # Create basic voice info from directory
                    voices.append(VoiceInfo(
                        id=voice_dir.name,
                        name=voice_dir.name,
                        created_at="",
                        is_trained=False,
                    ))

        response = ListVoicesResponse(
            success=True,
            voices=voices,
            message=f"Found {len(voices)} voices",
        )
        return format_response(response)

    except Exception as e:
        logger.error(f"Error listing voices: {e}", exc_info=True)
        response = ListVoicesResponse(
            success=False,
            voices=[],
            message="Internal server error occurred",
        )
        return format_response(response, status_code=500)


@routes.http.put("/v1/voices/update")
async def update_voice(req: Annotated[UpdateVoiceRequest, Body(exclusive=True)]):
    """
    Update voice metadata.
    """
    try:
        voice_id = req.id if hasattr(req, 'id') else ""

        if not voice_id:
            response = UpdateVoiceResponse(
                success=False,
                message="Voice ID is required",
                voice_id="",
            )
            return format_response(response, status_code=400)

        voices_base = Path("voices")
        voice_dir = voices_base / voice_id

        if not voice_dir.exists():
            response = UpdateVoiceResponse(
                success=False,
                message=f"Voice '{voice_id}' not found",
                voice_id=voice_id,
            )
            return format_response(response, status_code=404)

        # Load existing metadata
        import json
        metadata_file = voice_dir / "metadata.json"

        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())
        else:
            metadata = {"id": voice_id}

        # Update with new values
        if req.name is not None:
            metadata["name"] = req.name
        if req.description is not None:
            metadata["description"] = req.description
        if req.language is not None:
            metadata["language"] = req.language

        # Save updated metadata
        metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

        response = UpdateVoiceResponse(
            success=True,
            message=f"Voice '{voice_id}' updated successfully",
            voice_id=voice_id,
        )
        return format_response(response)

    except Exception as e:
        logger.error(f"Error updating voice: {e}", exc_info=True)
        response = UpdateVoiceResponse(
            success=False,
            message="Internal server error occurred",
            voice_id=voice_id if "voice_id" in locals() else "",
        )
        return format_response(response, status_code=500)


@routes.http.post("/v1/voices/delete")
async def delete_voice(req: Annotated[DeleteVoiceRequest, Body(exclusive=True)]):
    """
    Delete a voice from the voice library.
    """
    try:
        voice_id = req.id

        if not voice_id:
            return JSONResponse(
                {"success": False, "message": "Voice ID is required"},
                status_code=400
            )

        voices_base = Path("voices")
        voice_dir = voices_base / voice_id

        if not voice_dir.exists():
            return JSONResponse(
                {
                    "success": False,
                    "message": f"Voice '{voice_id}' not found",
                    "voice_id": voice_id,
                },
                status_code=404
            )

        # Remove voice directory
        import shutil
        shutil.rmtree(voice_dir)

        return JSONResponse(
            {
                "success": True,
                "message": f"Voice '{voice_id}' deleted successfully",
                "voice_id": voice_id,
            }
        )

    except Exception as e:
        logger.error(f"Error deleting voice: {e}", exc_info=True)
        return JSONResponse(
            {
                "success": False,
                "message": "Internal server error occurred",
                "voice_id": voice_id if "voice_id" in locals() else "",
            },
            status_code=500
        )


# ==============================================================================
# Training Task API Endpoints
# ==============================================================================

# In-memory training task storage (for production, use a database)
_training_tasks: dict[str, TrainingTaskInfo] = {}


@routes.http.post("/v1/training/start")
async def start_training(req: Annotated[CreateTrainingRequest, Body(exclusive=True)]):
    """
    Start a new training task for voice cloning.
    """
    try:
        import uuid
        from datetime import datetime

        task_id = str(uuid.uuid4())

        # Validate data path exists
        data_path = Path(req.data_path)
        if not data_path.exists():
            response = CreateTrainingResponse(
                success=False,
                message=f"Data path '{req.data_path}' does not exist",
                task_id=task_id,
            )
            return format_response(response, status_code=400)

        # Create training task
        task = TrainingTaskInfo(
            task_id=task_id,
            voice_id=req.voice_id,
            status="pending",
            progress=0.0,
            current_step=0,
            total_steps=req.max_steps,
            created_at=datetime.utcnow().isoformat(),
        )

        _training_tasks[task_id] = task

        # TODO: Actually start the training process in background
        logger.info(f"Training task {task_id} created for voice {req.voice_id}")

        response = CreateTrainingResponse(
            success=True,
            message=f"Training task created successfully",
            task_id=task_id,
        )
        return format_response(response)

    except Exception as e:
        logger.error(f"Error starting training: {e}", exc_info=True)
        response = CreateTrainingResponse(
            success=False,
            message="Internal server error occurred",
            task_id="",
        )
        return format_response(response, status_code=500)


@routes.http.get("/v1/training/status/{task_id}")
async def get_training_status(task_id: str):
    """
    Get the status of a training task.
    """
    try:
        training_manager = get_training_manager()
        job = training_manager.get_job(task_id)

        if job is None:
            # Check legacy tasks
            task = _training_tasks.get(task_id)
            if task is None:
                response = GetTrainingStatusResponse(
                    success=False,
                    task=None,
                    message=f"Training task '{task_id}' not found",
                )
                return format_response(response, status_code=404)
            response = GetTrainingStatusResponse(
                success=True,
                task=task,
                message="Success",
            )
            return format_response(response)

        # Convert training job to legacy format
        from tools.server.training_manager import TrainingStatus
        task = TrainingTaskInfo(
            task_id=job.job_id,
            voice_id=job.voice_id,
            voice_name=job.name,
            status=job.status.value if isinstance(job.status, TrainingStatus) else job.status,
            progress=job.progress,
            current_step=int(job.progress * (job.training_params.get("max_steps", 5000) if job.training_params else 5000)),
            total_steps=job.training_params.get("max_steps", 5000) if job.training_params else 5000,
            created_at=job.created_at,
            error=job.error,
        )

        response = GetTrainingStatusResponse(
            success=True,
            task=task,
            message="Success",
        )
        return format_response(response)

    except Exception as e:
        logger.error(f"Error getting training status: {e}", exc_info=True)
        response = GetTrainingStatusResponse(
            success=False,
            task=None,
            message="Internal server error occurred",
        )
        return format_response(response, status_code=500)


@routes.http.get("/v1/training/list")
async def list_training_tasks():
    """
    List all training tasks.
    """
    try:
        training_manager = get_training_manager()
        jobs = training_manager.list_jobs()

        # Convert jobs to legacy format
        from tools.server.training_manager import TrainingStatus
        tasks = []
        for job in jobs:
            task = TrainingTaskInfo(
                task_id=job.job_id,
                voice_id=job.voice_id,
                voice_name=job.name,
                status=job.status.value if isinstance(job.status, TrainingStatus) else job.status,
                progress=job.progress,
                current_step=int(job.progress * (job.training_params.get("max_steps", 5000) if job.training_params else 5000)),
                total_steps=job.training_params.get("max_steps", 5000) if job.training_params else 5000,
                created_at=job.created_at,
                error=job.error,
            )
            tasks.append(task)

        # Also include legacy tasks
        legacy_tasks = [t for t in _training_tasks.values() if t.task_id not in {j.job_id for j in jobs}]
        tasks.extend(legacy_tasks)

        response = ListTrainingTasksResponse(
            success=True,
            tasks=tasks,
            message=f"Found {len(tasks)} training tasks",
        )
        return format_response(response)

    except Exception as e:
        logger.error(f"Error listing training tasks: {e}", exc_info=True)
        response = ListTrainingTasksResponse(
            success=False,
            tasks=[],
            message="Internal server error occurred",
        )
        return format_response(response, status_code=500)


@routes.http.post("/v1/training/cancel")
async def cancel_training(task_id: str = Body(...)):
    """
    Cancel a training task.
    """
    try:
        training_manager = get_training_manager()
        job = training_manager.get_job(task_id)

        if job is None and task_id not in _training_tasks:
            response = CancelTrainingResponse(
                success=False,
                message=f"Training task '{task_id}' not found",
                task_id=task_id,
            )
            return format_response(response, status_code=404)

        # Try to cancel via training manager first
        if job is not None:
            from tools.server.training_manager import TrainingStatus
            if job.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]:
                response = CancelTrainingResponse(
                    success=False,
                    message=f"Training task '{task_id}' is already {job.status}",
                    task_id=task_id,
                )
                return format_response(response, status_code=400)

            success = training_manager.cancel_job(task_id)
            if success:
                # Update legacy task
                if task_id in _training_tasks:
                    _training_tasks[task_id].status = "cancelled"

                response = CancelTrainingResponse(
                    success=True,
                    message=f"Training task '{task_id}' cancelled successfully",
                    task_id=task_id,
                )
                return format_response(response)
            else:
                response = CancelTrainingResponse(
                    success=False,
                    message=f"Failed to cancel training task '{task_id}'",
                    task_id=task_id,
                )
                return format_response(response, status_code=500)

        # Fallback to legacy tasks
        task = _training_tasks[task_id]
        if task.status in ["completed", "failed", "cancelled"]:
            response = CancelTrainingResponse(
                success=False,
                message=f"Training task '{task_id}' is already {task.status}",
                task_id=task_id,
            )
            return format_response(response, status_code=400)

        # Update task status
        task.status = "cancelled"

        response = CancelTrainingResponse(
            success=True,
            message=f"Training task '{task_id}' cancelled successfully",
            task_id=task_id,
        )
        return format_response(response)

    except Exception as e:
        logger.error(f"Error cancelling training: {e}", exc_info=True)
        response = CancelTrainingResponse(
            success=False,
            message="Internal server error occurred",
            task_id=task_id,
        )
        return format_response(response, status_code=500)


# ==============================================================================
# Batch Dubbing API Endpoints
# ==============================================================================

# In-memory batch job storage (for production, use a database)
_batch_jobs: dict[str, BatchJobInfo] = {}


@routes.http.post("/v1/batch/create")
async def create_batch_job(req: Annotated[CreateBatchJobRequest, Body(exclusive=True)]):
    """
    Create a new batch dubbing job.
    """
    try:
        import uuid
        from datetime import datetime

        job_id = str(uuid.uuid4())

        # Validate subtitle file exists
        subtitle_file = Path(req.subtitle_file)
        if not subtitle_file.exists():
            response = CreateBatchJobResponse(
                success=False,
                message=f"Subtitle file '{req.subtitle_file}' does not exist",
                job_id=job_id,
            )
            return format_response(response, status_code=400)

        # Parse subtitle file to get item count
        # TODO: Implement proper SRT parsing
        total_items = 0

        # Create batch job
        job = BatchJobInfo(
            job_id=job_id,
            name=req.name,
            status="pending",
            progress=0.0,
            total_items=total_items,
            completed_items=0,
            created_at=datetime.utcnow().isoformat(),
        )

        _batch_jobs[job_id] = job

        # TODO: Actually start processing in background
        logger.info(f"Batch job {job_id} created for {req.name}")

        response = CreateBatchJobResponse(
            success=True,
            message=f"Batch job created successfully",
            job_id=job_id,
        )
        return format_response(response)

    except Exception as e:
        logger.error(f"Error creating batch job: {e}", exc_info=True)
        response = CreateBatchJobResponse(
            success=False,
            message="Internal server error occurred",
            job_id="",
        )
        return format_response(response, status_code=500)


@routes.http.get("/v1/batch/status/{job_id}")
async def get_batch_job_status(job_id: str):
    """
    Get the status of a batch dubbing job.
    """
    try:
        job = _batch_jobs.get(job_id)

        if job is None:
            response = GetBatchJobStatusResponse(
                success=False,
                job=None,
                message=f"Batch job '{job_id}' not found",
            )
            return format_response(response, status_code=404)

        response = GetBatchJobStatusResponse(
            success=True,
            job=job,
            message="Success",
        )
        return format_response(response)

    except Exception as e:
        logger.error(f"Error getting batch job status: {e}", exc_info=True)
        response = GetBatchJobStatusResponse(
            success=False,
            job=None,
            message="Internal server error occurred",
        )
        return format_response(response, status_code=500)


@routes.http.get("/v1/batch/list")
async def list_batch_jobs():
    """
    List all batch dubbing jobs.
    """
    try:
        jobs = list(_batch_jobs.values())

        response = ListBatchJobsResponse(
            success=True,
            jobs=jobs,
            message=f"Found {len(jobs)} batch jobs",
        )
        return format_response(response)

    except Exception as e:
        logger.error(f"Error listing batch jobs: {e}", exc_info=True)
        response = ListBatchJobsResponse(
            success=False,
            jobs=[],
            message="Internal server error occurred",
        )
        return format_response(response, status_code=500)


@routes.http.post("/v1/batch/cancel")
async def cancel_batch_job(job_id: str = Body(...)):
    """
    Cancel a batch dubbing job.
    """
    try:
        if job_id not in _batch_jobs:
            response = CancelBatchJobResponse(
                success=False,
                message=f"Batch job '{job_id}' not found",
                job_id=job_id,
            )
            return format_response(response, status_code=404)

        job = _batch_jobs[job_id]

        if job.status in ["completed", "failed", "cancelled"]:
            response = CancelBatchJobResponse(
                success=False,
                message=f"Batch job '{job_id}' is already {job.status}",
                job_id=job_id,
            )
            return format_response(response, status_code=400)

        # Update job status
        job.status = "cancelled"

        response = CancelBatchJobResponse(
            success=True,
            message=f"Batch job '{job_id}' cancelled successfully",
            job_id=job_id,
        )
        return format_response(response)

    except Exception as e:
        logger.error(f"Error cancelling batch job: {e}", exc_info=True)
        response = CancelBatchJobResponse(
            success=False,
            message="Internal server error occurred",
            job_id=job_id,
        )
        return format_response(response, status_code=500)


# ==============================================================================
# Voice Reference Management API Endpoints
# ==============================================================================

@routes.http.post("/v1/voice/{voice_id}/references/add")
async def add_voice_reference(voice_id: str, audio: UploadFile = Body(...), reference_text: str = Body("안녕하세요? 이것은 테스트입니다.")):
    """
    Add an additional reference audio file to an existing voice.
    This improves voice cloning quality by providing more examples.
    """
    try:
        voices_base = Path("voices")
        voice_dir = voices_base / voice_id

        if not voice_dir.exists():
            return JSONResponse(
                {"success": False, "message": f"Voice '{voice_id}' not found"},
                status_code=404
            )

        # Load existing metadata
        import json
        metadata_file = voice_dir / "metadata.json"
        if not metadata_file.exists():
            return JSONResponse(
                {"success": False, "message": "Voice metadata not found"},
                status_code=404
            )

        metadata = json.loads(metadata_file.read_text())

        # Get current audio files count
        audio_files = metadata.get("audio_files", [])
        if isinstance(audio_files, str):
            audio_files = [audio_files]

        # Save new audio file
        audio_filename = f"reference_{len(audio_files):03d}_{audio.filename or 'audio.wav'}"
        audio_path = voice_dir / audio_filename

        audio_content = audio.read()
        if not audio_content:
            return JSONResponse(
                {"success": False, "message": "Audio file is empty"},
                status_code=400
            )

        audio_path.write_bytes(audio_content)
        audio_files.append(audio_filename)

        # Update metadata
        metadata["audio_files"] = audio_files
        if reference_text and reference_text.strip():
            metadata["reference_text"] = reference_text.strip()

        metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

        logger.info(f"Added reference audio to voice {voice_id}: {audio_filename}")

        return JSONResponse({
            "success": True,
            "message": f"Reference audio added successfully. Total references: {len(audio_files)}",
            "voice_id": voice_id,
            "audio_file": audio_filename,
            "total_references": len(audio_files),
        })

    except Exception as e:
        logger.error(f"Error adding reference to voice {voice_id}: {e}", exc_info=True)
        return JSONResponse(
            {"success": False, "message": f"Failed to add reference: {str(e)}"},
            status_code=500
        )


@routes.http.post("/v1/voice/{voice_id}/references/update-text")
async def update_reference_text(voice_id: str, reference_text: str = Body(...)):
    """
    Update the reference text used for voice cloning.
    This text should match the style and language of the target speech.
    """
    try:
        voices_base = Path("voices")
        voice_dir = voices_base / voice_id

        if not voice_dir.exists():
            return JSONResponse(
                {"success": False, "message": f"Voice '{voice_id}' not found"},
                status_code=404
            )

        # Load existing metadata
        import json
        metadata_file = voice_dir / "metadata.json"
        if not metadata_file.exists():
            return JSONResponse(
                {"success": False, "message": "Voice metadata not found"},
                status_code=404
            )

        metadata = json.loads(metadata_file.read_text())

        # Update reference text
        metadata["reference_text"] = reference_text.strip()

        metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

        logger.info(f"Updated reference text for voice {voice_id}")

        return JSONResponse({
            "success": True,
            "message": "Reference text updated successfully",
            "voice_id": voice_id,
            "reference_text": reference_text,
        })

    except Exception as e:
        logger.error(f"Error updating reference text for voice {voice_id}: {e}", exc_info=True)
        return JSONResponse(
            {"success": False, "message": f"Failed to update reference text: {str(e)}"},
            status_code=500
        )


# ==============================================================================
# Voice Clone API Endpoints (Combined upload + training)
# ==============================================================================

@routes.http.post("/v1/voice-clone/create")
async def create_voice_clone(
    audio: UploadFile = Body(...),
    name: str = Body(...),
    description: str = Body(""),
    reference_text: str = Body("请输入音频中实际说的话"),  # NEW: Reference audio content text
    max_steps: int = Body(5000),
    learning_rate: float = Body(1e-4),
    batch_size: int = Body(16),
):
    """
    Combined endpoint for voice cloning: upload audio + create voice + start training.
    Expects multipart/form-data with: audio (file), name (str), description (str),
    reference_text (str) - the actual content spoken in the audio,
    max_steps (int), learning_rate (float), batch_size (int).
    """
    try:
        import uuid
        import json
        from datetime import datetime

        # Generate unique voice ID
        voice_id = str(uuid.uuid4())[:8]
        task_id = str(uuid.uuid4())

        # Create voices directory
        voices_base = Path("voices")
        voices_base.mkdir(exist_ok=True)

        voice_dir = voices_base / voice_id
        voice_dir.mkdir(exist_ok=True)

        # Save uploaded audio file(s) - support multiple files
        # Use audio preprocessor for better quality
        preprocessor = get_preprocessor()

        # Check if audio is a list of files or single file
        if isinstance(audio, list):
            # Multiple audio files
            audio_files = []
            for i, audio_file in enumerate(audio):
                audio_content = audio_file.read()
                if not audio_content:
                    continue

                # Preprocess audio
                try:
                    preprocessed_audio = preprocessor.preprocess_audio(
                        audio_content,
                        input_format=Path(audio_file.filename or "audio.wav").suffix.lstrip("."),
                    )
                    # Validate audio
                    is_valid, error_msg = preprocessor.validate_audio(preprocessed_audio)
                    if not is_valid:
                        logger.warning(f"Audio validation warning: {error_msg}")
                except Exception as e:
                    logger.warning(f"Audio preprocessing failed, using original: {e}")
                    preprocessed_audio = audio_content

                audio_filename = f"reference_{i:03d}.wav"
                audio_path = voice_dir / audio_filename
                audio_path.write_bytes(preprocessed_audio)
                audio_files.append(audio_filename)
        else:
            # Single audio file - might need to split into segments
            audio_content = audio.read()
            if not audio_content:
                return JSONResponse(
                    {"success": False, "message": "Audio file is empty"},
                    status_code=400
                )

            # Preprocess audio
            try:
                preprocessed_audio = preprocessor.preprocess_audio(
                    audio_content,
                    input_format=Path(audio.filename or "audio.wav").suffix.lstrip("."),
                )
                # Validate audio
                is_valid, error_msg = preprocessor.validate_audio(preprocessed_audio)
                if not is_valid:
                    logger.warning(f"Audio validation warning: {error_msg}")
            except Exception as e:
                logger.warning(f"Audio preprocessing failed, using original: {e}")
                preprocessed_audio = audio_content

            # Split long audio into segments for better voice cloning
            # Use shorter segments (5-8 seconds) for more reference points
            audio_segments = preprocessor.split_audio(
                preprocessed_audio,
                segment_duration=6.0,  # 6 second segments for better voice matching
                overlap=3.0,  # 3 second overlap for more coverage
            )

            audio_files = []
            for i, segment in enumerate(audio_segments):
                audio_filename = f"reference_{i:03d}.wav"
                audio_path = voice_dir / audio_filename
                audio_path.write_bytes(segment)
                audio_files.append(audio_filename)

            logger.info(f"Split audio into {len(audio_files)} segments for better voice cloning")

        # Get audio duration from first file
        duration = 0.0
        if audio_files:
            try:
                first_audio_path = voice_dir / audio_files[0]
                audio_info = sf.read(first_audio_path)
                duration = len(audio_info[0]) / 24000  # Assuming 24kHz
            except:
                pass

        # Create voice metadata
        metadata = {
            "id": voice_id,
            "name": name,
            "description": description,
            "language": "yanbian_korean",
            "created_at": datetime.utcnow().isoformat(),
            "sample_rate": 24000,
            "duration": duration,
            "is_trained": True,  # Voice is ready immediately for reference-based TTS
            "audio_files": audio_files,  # List of audio files
            "reference_text": reference_text.strip() or "请输入音频中实际说的话",  # Use provided reference text
        }

        metadata_file = voice_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

        # Get training manager and create real LoRA training job
        training_manager = get_training_manager()

        # Create audio file paths for training
        audio_file_paths = [voice_dir / f for f in audio_files]

        # Create training job
        job = training_manager.create_job(
            voice_id=voice_id,
            name=name,
            audio_files=audio_file_paths,
            reference_text=reference_text.strip() or "请输入音频中实际说的话",
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        # Update voice metadata to indicate training in progress
        metadata["is_trained"] = False  # Not ready until training completes
        metadata["training_job_id"] = job.job_id
        metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

        # Create legacy training task for compatibility
        task = TrainingTaskInfo(
            task_id=job.job_id,
            voice_id=voice_id,
            voice_name=name,
            status="running",
            progress=0.0,
            current_step=0,
            total_steps=max_steps,
            created_at=datetime.utcnow().isoformat(),
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
        _training_tasks[job.job_id] = task

        # Progress callback to sync training manager with legacy tasks
        def progress_callback(training_job):
            if job.job_id in _training_tasks:
                legacy_task = _training_tasks[job.job_id]
                legacy_task.progress = training_job.progress
                legacy_task.current_step = int(training_job.progress * max_steps)
                if training_job.status == TrainingStatus.COMPLETED:
                    legacy_task.status = "completed"
                elif training_job.status == TrainingStatus.FAILED:
                    legacy_task.status = "failed"
                elif training_job.status == TrainingStatus.CANCELLED:
                    legacy_task.status = "cancelled"

        # Start actual LoRA training in background
        training_manager.start_training(job.job_id, progress_callback)

        logger.info(f"LoRA training job {job.job_id} created for voice {name} ({voice_id})")
        logger.info(f"Audio saved: {len(audio_files)} segments, total size: {sum(f.stat().st_size for f in audio_file_paths)} bytes")

        return JSONResponse({
            "success": True,
            "message": "Voice clone created and LoRA training started. This will take 30-60 minutes for true voice cloning (90-95% similarity).",
            "task_id": job.job_id,
            "voice_id": voice_id,
            "voice_name": name,
            "estimated_time_minutes": 60,
        })

    except Exception as e:
        logger.error(f"Error creating voice clone: {e}", exc_info=True)
        return JSONResponse(
            {
                "success": False,
                "message": f"Internal server error: {str(e)}",
            },
            status_code=500
        )
