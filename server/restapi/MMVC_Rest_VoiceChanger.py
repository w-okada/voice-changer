import numpy as np
import msgpack

from fastapi import APIRouter, Request
from fastapi.responses import Response, PlainTextResponse
from const import get_edition, get_version
from voice_changer.VoiceChangerManager import VoiceChangerManager

import logging
logger = logging.getLogger(__name__)


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


    async def test(self, req: Request):
        try:
            data = await req.body()
            timestamp, voice = msgpack.unpackb(data)

            unpackedData = np.frombuffer(voice, dtype=np.int16).astype(np.float32) / 32768

            out_audio, perf, err = self.voiceChangerManager.changeVoice(unpackedData)
            out_audio = (out_audio * 32767).astype(np.int16).tobytes()

            if err is not None:
                error_code, error_message = err
                return Response(
                    content=msgpack.packb({
                        "error": True,
                        "timestamp": timestamp,
                        "details": {
                            "code": error_code,
                            "message": error_message,
                        },
                    }),
                    headers={'Content-Type': 'application/octet-stream'},
                )

            return Response(
                content=msgpack.packb({
                    "error": False,
                    "timestamp": timestamp,
                    "audio": out_audio,
                    "perf": perf,
                }),
                headers={'Content-Type': 'application/octet-stream'},
            )

        except Exception as e:
            logger.exception(e)
            return Response(
                content=msgpack.packb({
                    "error": True,
                    "timestamp": 0,
                    "details": {
                        "code": "GENERIC_REST_SERVER_ERROR",
                        "message": "Check command line for more details.",
                    },
                }),
                headers={'Content-Type': 'application/octet-stream'},
            )
