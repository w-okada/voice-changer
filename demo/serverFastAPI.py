import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import logging
import os, sys, base64, traceback, struct

import torch
import numpy as np
from scipy.io.wavfile import write, read

sys.path.append("mod")
sys.path.append("mod/text")

import utils
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols


class VoiceChanger():
    def __init__(self, config, model):
        self.hps =utils.get_hparams_from_file(config)
        self.net_g = SynthesizerTrn(
                len(symbols),
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers=self.hps.data.n_speakers,
                **self.hps.model)
        self.net_g.eval()
        self.gpu_num = torch.cuda.device_count()
        print("GPU_NUM:",self.gpu_num)
        utils.load_checkpoint( model, self.net_g, None)


    def on_request(self, gpu, srcId, dstId, timestamp, wav): 
        if wav==0:
            samplerate, data=read("dummy.wav")
            unpackedData = data
        else:
            unpackedData = np.array(struct.unpack('<%sh'%(len(wav) // struct.calcsize('<h') ), wav))
            write("logs/received_data.wav", 24000, unpackedData.astype(np.int16))

        try:
            if gpu<0 or self.gpu_num==0 :
                with torch.no_grad():
                    dataset = TextAudioSpeakerLoader("dummy.txt", self.hps.data, no_use_textfile=True)
                    data = dataset.get_audio_text_speaker_pair([ unpackedData, srcId, "a"])
                    data = TextAudioSpeakerCollate()([data])
                    x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [x.cpu() for x in data]
                    sid_tgt1 = torch.LongTensor([dstId]).cpu()
                    audio1 = (self.net_g.cpu().voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt1)[0][0,0].data * self.hps.data.max_wav_value).cpu().float().numpy()
            else:
                with torch.no_grad():
                    dataset = TextAudioSpeakerLoader("dummy.txt", self.hps.data, no_use_textfile=True)
                    data = dataset.get_audio_text_speaker_pair([ unpackedData, srcId, "a"])
                    data = TextAudioSpeakerCollate()([data])
                    x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [x.cuda(gpu) for x in data]
                    sid_tgt1 = torch.LongTensor([dstId]).cuda(gpu)
                    audio1 = (self.net_g.cuda(gpu).voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt1)[0][0,0].data * self.hps.data.max_wav_value).cpu().float().numpy()
        except Exception as e:
            print("VC PROCESSING!!!! EXCEPTION!!!", e)
            print(traceback.format_exc())
        
        audio1 = audio1.astype(np.int16)
        return audio1


logger = logging.getLogger('uvicorn')  

args = sys.argv
PORT = args[1]
CONFIG = args[2]
MODEL  = args[3]
logger.info('INITIALIZE MODEL')
voiceChanger = VoiceChanger(CONFIG, MODEL)
voiceChanger.on_request(0,0,0,0,0)



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/front", StaticFiles(directory="../frontend/dist", html=True), name="static")

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
    global voiceChanger
    try:
        print("POST REQUEST PROCESSING....")
        gpu = voice.gpu
        srcId = voice.srcId
        dstId = voice.dstId
        timestamp = voice.timestamp
        buffer = voice.buffer
        wav = base64.b64decode(buffer)

        changedVoice = voiceChanger.on_request(gpu, srcId, dstId, timestamp, wav)
        changedVoiceBase64 = base64.b64encode(changedVoice).decode('utf-8')

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
    logger.info('START APP')
    uvicorn.run(f"{os.path.basename(__file__)[:-3]}:app", host="0.0.0.0", port=int(PORT), reload=True, log_level="info")
