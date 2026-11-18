import base64
import struct
import numpy as np
import traceback

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from voice_changer.VoiceChangerManager import VoiceChangerManager
from pydantic import BaseModel
import threading


class VoiceModel(BaseModel):
    timestamp: int
    buffer: str


class ConvertModel(BaseModel):
    sampleRate: int  # buffer のサンプリングレート (8000-96000)
    buffer: str  # base64 (16bit LE mono PCM)


CONVERT_MAX_SEC = 120


class MMVC_Rest_VoiceChanger:
    def __init__(self, voiceChangerManager: VoiceChangerManager):
        self.voiceChangerManager = voiceChangerManager
        self.router = APIRouter()
        self.router.add_api_route("/test", self.test, methods=["POST"])
        self.router.add_api_route("/convert", self.convert, methods=["POST"])

        self.tlock = threading.Lock()
        self.convert_lock = threading.Lock()  # 一括変換どうしを直列化 (リアルタイム用 tlock とは独立)

    def convert(self, req: ConvertModel):
        """文全体を1回で変換する (リアルタイムのチャンク処理を通さない)。TTS連携向け。"""
        if not 8000 <= req.sampleRate <= 96000:
            raise HTTPException(status_code=400, detail="sampleRate must be 8000-96000")
        try:
            raw = base64.b64decode(req.buffer, validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail="buffer is not valid base64")
        if len(raw) == 0 or len(raw) % 2 != 0:
            raise HTTPException(status_code=400, detail="buffer must be non-empty 16bit PCM")
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.int16)
        if pcm.shape[0] > CONVERT_MAX_SEC * req.sampleRate:
            raise HTTPException(status_code=413, detail=f"audio too long (max {CONVERT_MAX_SEC}s)")

        # 同時実行はGPUメモリを圧迫するので待たせず断る
        if not self.convert_lock.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="another conversion is running")
        try:
            out, out_sr = self.voiceChangerManager.convert_offline(pcm, req.sampleRate)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            print("CONVERT EXCEPTION:", e)
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            self.convert_lock.release()

        return JSONResponse(content={"sampleRate": int(out_sr), "buffer": base64.b64encode(out.astype("<i2").tobytes()).decode("ascii")})

    def test(self, voice: VoiceModel):
        try:
            timestamp = voice.timestamp
            buffer = voice.buffer
            wav = base64.b64decode(buffer)

            # if wav == 0:
            #     samplerate, data = read("dummy.wav")
            #     unpackedData = data
            # else:
            #     unpackedData = np.array(
            #         struct.unpack("<%sh" % (len(wav) // struct.calcsize("<h")), wav)
            #     )

            unpackedData = np.array(struct.unpack("<%sh" % (len(wav) // struct.calcsize("<h")), wav)).astype(np.int16)
            # print(f"[REST] unpackedDataType {unpackedData.dtype}")

            self.tlock.acquire()
            changedVoice = self.voiceChangerManager.changeVoice(unpackedData)
            self.tlock.release()

            changedVoiceBase64 = base64.b64encode(changedVoice[0]).decode("utf-8")
            data = {"timestamp": timestamp, "changedVoiceBase64": changedVoiceBase64}

            json_compatible_item_data = jsonable_encoder(data)
            return JSONResponse(content=json_compatible_item_data)

        except Exception as e:
            print("REQUEST PROCESSING!!!! EXCEPTION!!!", e)
            print(traceback.format_exc())
            self.tlock.release()
            return str(e)
