import torch

from scipy.io.wavfile import write, read
import numpy as np
import traceback

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
        utils.load_checkpoint(model, self.net_g, None)

        text_norm = text_to_sequence("a", self.hps.data.text_cleaners)
        text_norm = commons.intersperse(text_norm, 0)
        self.text_norm = torch.LongTensor(text_norm)
        self.audio_buffer = torch.zeros(1, 0)
        self.prev_audio = np.zeros(1)
        self.mps_enabled = getattr(
            torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

        print(
            f"VoiceChanger Initialized (GPU_NUM:{self.gpu_num}, mps_enabled:{self.mps_enabled})")

    def destroy(self):
        del self.net_g

    def on_request(self, gpu, srcId, dstId, timestamp, prefixChunkSize, wav):
        unpackedData = wav
        convertSize = unpackedData.shape[0] + (prefixChunkSize * 512)

        try:

            audio = torch.FloatTensor(unpackedData.astype(np.float32))
            audio_norm = audio / self.hps.data.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            self.audio_buffer = torch.cat(
                [self.audio_buffer, audio_norm], axis=1)
            audio_norm = self.audio_buffer[:, -convertSize:]
            self.audio_buffer = audio_norm

            spec = spectrogram_torch(audio_norm, self.hps.data.filter_length,
                                     self.hps.data.sampling_rate, self.hps.data.hop_length, self.hps.data.win_length,
                                     center=False)
            spec = torch.squeeze(spec, 0)
            sid = torch.LongTensor([int(srcId)])

            data = (self.text_norm, spec, audio_norm, sid)
            data = TextAudioSpeakerCollate()([data])

            # if gpu < 0 or (self.gpu_num == 0 and not self.mps_enabled):
            if gpu < 0 or self.gpu_num == 0:
                with torch.no_grad():
                    x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [
                        x.cpu() for x in data]
                    sid_tgt1 = torch.LongTensor([dstId]).cpu()
                    audio1 = (self.net_g.cpu().voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt1)[
                              0][0, 0].data * self.hps.data.max_wav_value).cpu().float().numpy()
            # elif self.mps_enabled == True: # MPS doesnt support aten::weight_norm_interface, and PYTORCH_ENABLE_MPS_FALLBACK=1 cause a big dely.
            #         x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [
            #             x.to("mps") for x in data]
            #         sid_tgt1 = torch.LongTensor([dstId]).to("mps")
            #         audio1 = (self.net_g.to("mps").voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt1)[
            #                   0][0, 0].data * self.hps.data.max_wav_value).cpu().float().numpy()

            else:
                with torch.no_grad():
                    x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [
                        x.cuda(gpu) for x in data]
                    sid_tgt1 = torch.LongTensor([dstId]).cuda(gpu)
                    audio1 = (self.net_g.cuda(gpu).voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt1)[
                              0][0, 0].data * self.hps.data.max_wav_value).cpu().float().numpy()

            # if len(self.prev_audio) > unpackedData.shape[0]:
            #     prevLastFragment = self.prev_audio[-unpackedData.shape[0]:]
            #     curSecondLastFragment = audio1[-unpackedData.shape[0]*2:-unpackedData.shape[0]]
            #     print("prev, cur", prevLastFragment.shape, curSecondLastFragment.shape)
            # self.prev_audio = audio1
            # print("self.prev_audio", self.prev_audio.shape)

            audio1 = audio1[-unpackedData.shape[0]*2:]

        except Exception as e:
            print("VC PROCESSING!!!! EXCEPTION!!!", e)
            print(traceback.format_exc())

        audio1 = audio1.astype(np.int16)
        return audio1
