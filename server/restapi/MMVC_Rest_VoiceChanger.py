import base64
import numpy as np
import traceback
import os

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, PlainTextResponse
from const import get_edition, get_version
from voice_changer.VoiceChangerManager import VoiceChangerManager
from pydantic import BaseModel
import threading


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

        self.tlock = threading.Lock()


    def edition(self):
        return PlainTextResponse(get_edition())


    def version(self):
        return PlainTextResponse(get_version())


    def test(self, voice: VoiceModel):
        try:
            data = base64.b64decode(voice.buffer)

            unpackedData = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768

            self.tlock.acquire()
            out_audio, perf = self.voiceChangerManager.changeVoice(unpackedData)
            self.tlock.release()
            out_audio = (out_audio * 32767).astype(np.int16).tobytes()

            return JSONResponse(content=jsonable_encoder({
                "timestamp": voice.timestamp,
                "changedVoiceBase64": base64.b64encode(out_audio).decode("utf-8"),
                "perf": perf
            }))

        except Exception as e:
            print("REQUEST PROCESSING!!!! EXCEPTION!!!", e)
            print(traceback.format_exc())
            self.tlock.release()
            return str(e)
