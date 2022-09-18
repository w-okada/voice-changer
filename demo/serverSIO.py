import eventlet
import socketio
import sys
from datetime import datetime
import struct

import torch
import numpy as np
from scipy.io.wavfile import write

sys.path.append("mod")
sys.path.append("mod/text")
import utils
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols

class MyCustomNamespace(socketio.Namespace): 
    def __init__(self, namespace, config, model):
        super().__init__(namespace)
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

    def on_connect(self, sid, environ):
        print('[{}] connet sid : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S') , sid))
        # print('[{}] connet env : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S') , environ))

    def on_request_message(self, sid, msg): 
        # print("MESSGaa", msg)
        gpu = int(msg[0])
        srcId = int(msg[1])
        dstId = int(msg[2])
        timestamp = int(msg[3])
        data = msg[4]
        # print(srcId, dstId, timestamp)
        unpackedData = np.array(struct.unpack('<%sh'%(len(data) // struct.calcsize('<h') ), data))
        write("logs/received_data.wav", 24000, unpackedData.astype(np.int16))

        # self.emit('response', msg)

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

        audio1 = audio1.astype(np.int16)
        bin = struct.pack('<%sh'%len(audio1), *audio1)

        # print("return timestamp", timestamp)        
        self.emit('response',[timestamp, bin])




    def on_disconnect(self, sid):
        # print('[{}] disconnect'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        pass;


if __name__ == '__main__':
    args = sys.argv
    PORT = args[1]
    CONFIG = args[2]
    MODEL  = args[3]
    print(f"start... PORT:{PORT}, CONFIG:{CONFIG}, MODEL:{MODEL}")    
    # sio = socketio.Server(cors_allowed_origins='http://localhost:8080') 
    sio = socketio.Server(cors_allowed_origins='*') 
    sio.register_namespace(MyCustomNamespace('/test', CONFIG, MODEL)) 
    app = socketio.WSGIApp(sio,static_files={
        '': '../frontend/dist',
        '/': '../frontend/dist/index.html',
    }) 
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0',int(PORT))), app) 
    