import eventlet
import socketio
import sys, math, base64
from datetime import datetime
import struct

import torch, torchaudio
import numpy as np
from scipy.io.wavfile import write, read


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


class MyCustomNamespace(socketio.Namespace): 
    def __init__(self, namespace):
        super().__init__(namespace)

    def on_connect(self, sid, environ):
        print('[{}] connet sid : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S') , sid))

    def on_request_message(self, sid, msg): 
        print("Processing Request...")
        gpu = int(msg[0])
        srcId = int(msg[1])
        dstId = int(msg[2])
        timestamp = int(msg[3])
        data = msg[4]
        # print(srcId, dstId, timestamp)
        unpackedData = np.array(struct.unpack('<%sh'%(len(data) // struct.calcsize('<h') ), data))
        write("logs/received_data.wav", 24000, unpackedData.astype(np.int16))

        source, sr = torchaudio.load("logs/received_data.wav") # デフォルトでnormalize=Trueがついており、float32に変換して読んでくれるらしいのでこれを使う。https://pytorch.org/audio/stable/backend.html

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
        # write("logs/converted_data.wav", 24000, arr)
        # changedVoiceBase64 = base64.b64encode(arr).decode('utf-8')

        # data = {
        #     "gpu":gpu,
        #     "srcId":srcId,
        #     "dstId":dstId,
        #     "timestamp":timestamp,
        #     "changedVoiceBase64":changedVoiceBase64
        # }

        # audio1 = audio1.astype(np.int16)
        bin = struct.pack('<%sh'%len(arr), *arr)

        # print("return timestamp", timestamp)        
        self.emit('response',[timestamp, bin])

    def on_disconnect(self, sid):
        # print('[{}] disconnect'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        pass;


if __name__ == '__main__':
    args = sys.argv
    PORT = args[1]
    print(f"start... PORT:{PORT}")    
    sio = socketio.Server(cors_allowed_origins='*') 
    sio.register_namespace(MyCustomNamespace('/test')) 
    app = socketio.WSGIApp(sio,static_files={
        '': '../frontend/dist',
        '/': '../frontend/dist/index.html',
    }) 
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0',int(PORT))), app) 
    