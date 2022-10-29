import torch
from scipy.io.wavfile import write, read
import numpy as np
import struct, traceback

import utils 
import commons 
from models import SynthesizerTrn
from text.symbols import symbols
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from mel_processing import spectrogram_torch
from text import text_to_sequence, cleaned_text_to_sequence


class VoiceChanger():
    def __init__(self, config, model):
        self.hps = utils.get_hparams_from_file(config)
        self.net_g = SynthesizerTrn(
                len(symbols),
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers=self.hps.data.n_speakers,
                **self.hps.model)
        self.net_g.eval()
        self.gpu_num = torch.cuda.device_count()
        utils.load_checkpoint( model, self.net_g, None)
        print(f"VoiceChanger Initialized (GPU_NUM:{self.gpu_num})")

    def destroy(self):
        del self.net_g

    def on_request(self, gpu, srcId, dstId, timestamp, wav): 
        # if wav==0:
        #     samplerate, data=read("dummy.wav")
        #     unpackedData = data
        # else:
        #     unpackedData = np.array(struct.unpack('<%sh'%(len(wav) // struct.calcsize('<h') ), wav))
        #     write("logs/received_data.wav", 24000, unpackedData.astype(np.int16))

        unpackedData = wav

        try:

            text_norm = text_to_sequence("a", self.hps.data.text_cleaners)
            text_norm = commons.intersperse(text_norm, 0)
            text_norm = torch.LongTensor(text_norm)

            audio = torch.FloatTensor(unpackedData.astype(np.float32))
            audio_norm = audio /self.hps.data.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)

            spec = spectrogram_torch(audio_norm, self.hps.data.filter_length,
                    self.hps.data.sampling_rate, self.hps.data.hop_length, self.hps.data.win_length,
                    center=False)
            spec = torch.squeeze(spec, 0)
            sid = torch.LongTensor([int(srcId)])
            
            data =  (text_norm, spec, audio_norm, sid)
            data = TextAudioSpeakerCollate()([data])

            if gpu<0 or self.gpu_num==0 :
                with torch.no_grad():
                    x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [x.cpu() for x in data]
                    sid_tgt1 = torch.LongTensor([dstId]).cpu()
                    audio1 = (self.net_g.cpu().voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt1)[0][0,0].data * self.hps.data.max_wav_value).cpu().float().numpy()
            else:
                with torch.no_grad():
                    x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [x.cuda(gpu) for x in data]
                    sid_tgt1 = torch.LongTensor([dstId]).cuda(gpu)
                    audio1 = (self.net_g.cuda(gpu).voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt1)[0][0,0].data * self.hps.data.max_wav_value).cpu().float().numpy()
        except Exception as e:
            print("VC PROCESSING!!!! EXCEPTION!!!", e)
            print(traceback.format_exc())
        
        audio1 = audio1.astype(np.int16)
        return audio1