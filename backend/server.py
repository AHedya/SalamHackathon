from typing import Union, Annotated
import ormsgpack

from utils import TTSRequest, load_reference
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fish_speech.utils.schema import ServeTTSRequest


import requests

app = FastAPI()

TTS_URL = "http://localhost:8080/v1/tts"

from typing import List, Optional

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Adjust if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/TTS/")
def get_synthesized_audio(text: str, reference_id: Union[str, None] = None):
    data_load = {"text": text}

    if reference_id:
        data_load["references"] = load_reference(reference_id)

    data_load["seed"] = 95

    tts_load = TTSRequest(**data_load)
    TTS_data_load = ServeTTSRequest(**tts_load.model_dump())
    response = requests.post(
        TTS_URL,
        data=ormsgpack.packb(TTS_data_load, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
        headers={
            "content-type": "application/msgpack",
        },
    )
    if response.status_code == 200:
        return Response(content=response.content, media_type="audio/mpeg")
    return {"message": "error handling tts server"}, 400
