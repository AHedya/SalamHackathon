from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest


import os

working_dir = os.path.dirname(__file__)


class TTSRequest(BaseModel):

    text: str = Field(..., description="Text to be synthesized")
    references: List | None = []

    reference_id: Optional[str] = Field(None, description="ID of the reference model")

    normalize: bool = Field(default=True, description="Normalize the audio output")
    format: str = Field(
        default="wav", choices=["wav", "mp3", "flac"], description="Audio format"
    )
    latency: str = Field(
        default="normal", choices=["normal", "balanced"], description="Latency setting"
    )
    max_new_tokens: int = Field(
        default=1024, description="Max new tokens to generate (0 means no limit)"
    )
    chunk_length: int = Field(default=200, description="Chunk length for synthesis")
    top_p: float = Field(default=0.7, description="Top-p sampling for synthesis")
    repetition_penalty: float = Field(
        default=1.2, description="Repetition penalty for synthesis"
    )
    temperature: float = Field(default=0.7, description="Temperature for sampling")
    streaming: bool = Field(default=False, description="Enable streaming response")
    use_memory_cache: str = Field(
        default="off", choices=["on", "off"], description="Memory cache setting"
    )

    seed: Optional[int] = Field(
        None, description="Random seed for inference (None means randomized)"
    )


def load_reference(ref_id: str) -> Tuple[Tuple[str], Tuple[str]]:
    """given reference id, return corresponding ref_audio path and ref_text

    Args:
        ref_id (str): refrence id

    Returns:
        Tuple[Tuple[str],Tuple[str]]: Tuple of corresponding audio path, and ref text
    """
    audio_ref = []
    text_ref = []
    for i in os.listdir(os.path.join(working_dir, "references", ref_id)):
        if i.endswith(".wav"):
            audio_ref.append(os.path.join(working_dir, "references", ref_id, i))
        else:
            with open(
                os.path.join(working_dir, "references", ref_id, i),
                "r",
                encoding="utf-8",
            ) as f:
                text_ref.append(f.readline())

    audio_ref = [audio_to_bytes(audio) for audio in audio_ref]

    res = [
        ServeReferenceAudio(
            audio=ref_audio if ref_audio is not None else b"", text=ref_text
        )
        for ref_text, ref_audio in zip(text_ref, audio_ref)
    ]
    return res


def audio_to_bytes(file_path):
    if not file_path:
        return None
    with open(file_path, "rb") as wav_file:
        wav = wav_file.read()
    return wav


if __name__ == "__main__":
    print(load_reference("1"))
