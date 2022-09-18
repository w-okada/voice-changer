import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import logging
from logging.config import dictConfig
import os, sys, math, base64, struct, traceback, time

import torch, torchaudio
import numpy as np
from scipy.io.wavfile import write, read
from datetime import datetime

args = sys.argv
PORT = args[1]
MODE = args[2]


logger = logging.getLogger('uvicorn')  
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if MODE == "colab":
    print("ENV: colab")
    app.mount("/front", StaticFiles(directory="../frontend/dist", html=True), name="static")
    
    hubert_model = torch.hub.load("bshall/hubert:main", "hubert_soft").cuda()
    acoustic_model = torch.hub.load("bshall/acoustic-model:main", "hubert_soft").cuda()
    hifigan_model = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_soft").cuda()
else:    
    print("ENV: Docker")

    app.mount("/front", StaticFiles(directory="../frontend/dist", html=True), name="static")

    sys.path.append("/hubert")
    from hubert import hubert_discrete, hubert_soft, kmeans100

    sys.path.append("/acoustic-model")
    from acoustic import hubert_discrete, hubert_soft

    sys.path.append("/hifigan")
    from hifigan import hifigan

    hubert_model = torch.load("/models/bshall_hubert_main.pt").cuda()
    acoustic_model = torch.load("/models/bshall_acoustic-model_main.pt").cuda()
    hifigan_model = torch.load("/models/bshall_hifigan_main.pt").cuda()


def applyVol(i, chunk, vols):
  curVol = vols[i] / 2
  if curVol < 0.0001:
    line = torch.zeros(chunk.size())
  else:
    line = torch.ones(chunk.size())

  volApplied = torch.mul(line, chunk)  
  volApplied = volApplied.unsqueeze(0)
  return volApplied


@app.get("/test")
def get_test():
    try:
        return request.args.get('query', '')
    except Exception as e:
        print("REQUEST PROCESSING!!!! EXCEPTION!!!", e)
        print(traceback.format_exc())
        return str(e)



class VoiceModel(BaseModel):
    gpu: int
    srcId: int
    dstId: int
    timestamp: int
    buffer: str

@app.post("/test")
def post_test(voice:VoiceModel):
    try:
        print("POST REQUEST PROCESSING....")
        gpu = voice.gpu
        srcId = voice.srcId
        dstId = voice.dstId
        timestamp = voice.timestamp
        buffer = voice.buffer
        wav = base64.b64decode(buffer)
        unpackedData = np.array(struct.unpack('<%sh'%(len(wav) // struct.calcsize('<h') ), wav))
        # received_data_file = f"received_data_{timestamp}.wav"
        received_data_file = "received_data.wav"
        write(received_data_file, 24000, unpackedData.astype(np.int16))
        source, sr = torchaudio.load(received_data_file) # デフォルトでnormalize=Trueがついており、float32に変換して読んでくれるらしいのでこれを使う。https://pytorch.org/audio/stable/backend.html

        source_16k = torchaudio.functional.resample(source, 24000, 16000)
        source_16k = source_16k.unsqueeze(0).cuda()
        # SOFT-VC
        with torch.inference_mode():
            units = hubert_model.units(source_16k)
            mel = acoustic_model.generate(units).transpose(1, 2)
            target = hifigan_model(mel)

        dest = torchaudio.functional.resample(target, 16000,24000)
        dest = dest.squeeze().cpu()

        # ソースの音量取得
        source = source.cpu()
        specgram = torchaudio.transforms.MelSpectrogram(sample_rate=24000)(source)
        vol_apply_window_size = math.ceil(len(source[0]) / specgram.size()[2])
        specgram = specgram.transpose(1,2)
        vols = [ torch.max(i) for i in specgram[0]]
        chunks = torch.split(dest, vol_apply_window_size,0)

        chunks = [applyVol(i,c,vols) for i, c in enumerate(chunks)]
        dest = torch.cat(chunks,1)
        arr = np.array(dest.squeeze())

        int_size = 2**(16 - 1) - 1
        arr = (arr * int_size).astype(np.int16)
        write("converted_data.wav", 24000, arr)
        changedVoiceBase64 = base64.b64encode(arr).decode('utf-8')

        data = {
            "gpu":gpu,
            "srcId":srcId,
            "dstId":dstId,
            "timestamp":timestamp,
            "changedVoiceBase64":changedVoiceBase64
        }

        json_compatible_item_data = jsonable_encoder(data)

        
        return JSONResponse(content=json_compatible_item_data)
    except Exception as e:
        print("REQUEST PROCESSING!!!! EXCEPTION!!!", e)
        print(traceback.format_exc())
        return str(e)


if __name__ == '__main__':
    args = sys.argv
    PORT = args[1]
    MODE = args[2]
    logger.info('INITIALIZE MODEL')
    logger.info('START APP')
    uvicorn.run(f"{os.path.basename(__file__)[:-3]}:app", host="0.0.0.0", port=int(PORT), reload=True, log_level="info")
