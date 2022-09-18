from flask import Flask, request, Markup, abort, jsonify, send_from_directory
from flask_cors import CORS
import logging
from logging.config import dictConfig
import sys
import base64

import torch
import numpy as np
from scipy.io.wavfile import write, read
from datetime import datetime

import traceback
import struct

sys.path.append("mod")
sys.path.append("mod/text")

import utils
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

app = Flask(__name__)
@app.route("/<path:path>")
def static_dir(path):
    return send_from_directory("../frontend/dist", path)

@app.route('/', methods=['GET'])
def redirect_to_index():
    return send_from_directory("../frontend/dist", 'index.html')

CORS(app, resources={r"/*": {"origins": "*"}}) 

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



@app.route('/test', methods=['GET', 'POST'])
def test():
    try:
        if request.method == 'GET':
            return request.args.get('query', '')
        elif request.method == 'POST':
            print("POST REQUEST PROCESSING....")
            gpu = int(request.json['gpu'])
            srcId = int(request.json['srcId'])
            dstId = int(request.json['dstId'])
            timestamp = int(request.json['timestamp'])
            buffer = request.json['buffer']
            wav = base64.b64decode(buffer)
            # print(wav)
            # print(base64.b64encode(wav))
            changedVoice = voiceChanger.on_request(gpu, srcId, dstId, timestamp, wav)
            changedVoiceBase64 = base64.b64encode(changedVoice).decode('utf-8')
            # print("changedVoice",changedVoice)
            # print("CV64",changedVoiceBase64)
            data = {
                "gpu":gpu,
                "srcId":srcId,
                "dstId":dstId,
                "timestamp":timestamp,
                "changedVoiceBase64":changedVoiceBase64
            }
            return jsonify(data)
        else:
            return abort(400)
    except Exception as e:
        print("REQUEST PROCESSING!!!! EXCEPTION!!!", e)
        print(traceback.format_exc())
        return str(e)


if __name__ == '__main__':
    args = sys.argv
    PORT = args[1]
    CONFIG = args[2]
    MODEL  = args[3]
    app.logger.info('INITIALIZE MODEL')
    voiceChanger = VoiceChanger(CONFIG, MODEL)
    voiceChanger.on_request(0,0,0,0,0)
    app.logger.info('START APP')
    app.run(debug=True, host='0.0.0.0',port=PORT)