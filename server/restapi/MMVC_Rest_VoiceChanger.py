import base64
import numpy as np

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, PlainTextResponse
from const import get_edition, get_version
from voice_changer.VoiceChangerManager import VoiceChangerManager
from pydantic import BaseModel

import logging
logger = logging.getLogger(__name__)

class VoiceModel(BaseModel):
    timestamp: int
    buffer: str


class MMVC_Rest_VoiceChanger:
    def __init__(self, voiceChangerManager: VoiceChangerManager):
        self.voiceChangerManager = voiceChangerManager
        self.router = APIRouter()
        self.router.add_api_route("/test", self.test, methods=["POST"])
        self.router.add_api_route("/edition", self.edition, methods=["GET"])
        self.router.add_api_route("/version", self.version, methods=["GET"])


    def edition(self):
        return PlainTextResponse(get_edition())


    def version(self):
        return PlainTextResponse(get_version())


    def test(self, voice: VoiceModel):
        try:
            data = base64.b64decode(voice.buffer)

            unpackedData = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768

            out_audio, perf, err = self.voiceChangerManager.changeVoice(unpackedData)
            out_audio = (out_audio * 32767).astype(np.int16).tobytes()

            if err is not None:
                error_code, error_message = err
                return JSONResponse(content=jsonable_encoder({
                    "error": True,
                    "timestamp": voice.timestamp,
                    "details": {
                        "code": error_code,
                        "message": error_message,
                    },
                }))
            else:
                return JSONResponse(content=jsonable_encoder({
                    "error": False,
                    "timestamp": voice.timestamp,
                    "changedVoiceBase64": base64.b64encode(out_audio).decode("utf-8"),
                    "perf": perf,
                }))

        except Exception as e:
            logger.exception(e)
            return JSONResponse(content=jsonable_encoder({
                "error": True,
                "timestamp": 0,
                "details": {
                    "code": "GENERIC_REST_SERVER_ERROR",
                    "message": "Check command line for more details.",
                },
            }))
