import base64, struct
import numpy as np

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from voice_changer.VoiceChangerManager import VoiceChangerManager
from pydantic import BaseModel
import threading

class VoiceModel(BaseModel):
    gpu: int
    srcId: int
    dstId: int
    timestamp: int
    convertChunkNum: int
    crossFadeLowerValue: float
    crossFadeOffsetRate:float
    crossFadeEndRate:float
    buffer: str

class MMVC_Rest_VoiceChanger:
    def __init__(self, voiceChangerManager:VoiceChangerManager):
        self.voiceChangerManager = voiceChangerManager
        self.router = APIRouter()
        self.router.add_api_route("/test", self.test, methods=["POST"])
        self.tlock = threading.Lock()


    def test(self, voice: VoiceModel):
        try:
            gpu = voice.gpu
            srcId = voice.srcId
            dstId = voice.dstId
            timestamp = voice.timestamp
            convertChunkNum = voice.convertChunkNum
            crossFadeLowerValue = voice.crossFadeLowerValue
            crossFadeOffsetRate = voice.crossFadeOffsetRate
            crossFadeEndRate = voice.crossFadeEndRate
            buffer = voice.buffer
            wav = base64.b64decode(buffer)

            if wav == 0:
                samplerate, data = read("dummy.wav")
                unpackedData = data
            else:
                unpackedData = np.array(struct.unpack(
                    '<%sh' % (len(wav) // struct.calcsize('<h')), wav))
                # write("logs/received_data.wav", 24000,
                #       unpackedData.astype(np.int16))

            self.tlock.acquire()
            changedVoice = self.voiceChangerManager.changeVoice(
                gpu, srcId, dstId, timestamp, convertChunkNum, crossFadeLowerValue, crossFadeOffsetRate, crossFadeEndRate, unpackedData)
            self.tlock.release()

            changedVoiceBase64 = base64.b64encode(changedVoice).decode('utf-8')
            data = {
                "gpu": gpu,
                "srcId": srcId,
                "dstId": dstId,
                "timestamp": timestamp,
                "convertChunkNum": voice.convertChunkNum,
                "changedVoiceBase64": changedVoiceBase64
            }

            json_compatible_item_data = jsonable_encoder(data)
            return JSONResponse(content=json_compatible_item_data)

        except Exception as e:
            print("REQUEST PROCESSING!!!! EXCEPTION!!!", e)
            print(traceback.format_exc())
            return str(e)



