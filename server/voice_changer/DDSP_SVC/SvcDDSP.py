# original from: https://raw.githubusercontent.com/yxlllc/DDSP-SVC/master/gui_diff.py

import torch

try:
    from .models.ddsp.vocoder import load_model, F0_Extractor, Volume_Extractor, Units_Encoder
except Exception as e:
    print(e)
    from .models.ddsp.vocoder import load_model, F0_Extractor, Volume_Extractor, Units_Encoder

from .models.ddsp.core import upsample
from .models.enhancer import Enhancer
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
import numpy as np
from .models.diffusion.infer_gt_mel import DiffGtMel

class SvcDDSP:
    def __init__(self) -> None:
        self.ddsp_model = None
        self.ddsp_args = None
        self.diff_model = DiffGtMel()
        self.args = None
        self.units_encoder = None
        self.encoder_type = None
        self.encoder_ckpt = None
        self.k_step_max = 1000

    def setVCParams(self, params: VoiceChangerParams):
        self.params = params

    def update_model(self, ddsp_model_path: str, diff_model_path: str, device: torch.device):
        self.device = device
        
        print(ddsp_model_path, diff_model_path)
        # load ddsp model
        if self.ddsp_model is None or self.ddsp_model_path != ddsp_model_path:
            if ddsp_model_path is not None and ddsp_model_path != 'builtin':
                self.ddsp_model, self.ddsp_args = load_model(ddsp_model_path, device=self.device)
                self.ddsp_model_path = ddsp_model_path
        
        # load diffusion model
        self.diff_model.flush_model(diff_model_path, ddsp_config=self.ddsp_args)
        self.args = self.diff_model.args
        if self.args.model.type == 'DiffusionNew':
            self.k_step_max = self.args.model.k_step_max
            
        # load units encoder
        if self.units_encoder is None or self.args.data.encoder != self.encoder_type or self.args.data.encoder_ckpt != self.encoder_ckpt:
            if self.args.data.encoder == 'cnhubertsoftfish':
                cnhubertsoft_gate = self.args.data.cnhubertsoft_gate
            else:
                cnhubertsoft_gate = 10

            if self.args.data.encoder == "hubertsoft":
                encoderPath = self.params.hubert_soft
            elif self.args.data.encoder == "hubertbase":
                encoderPath = self.params.hubert_base
            elif self.args.data.encoder == "hubertbase768":
                encoderPath = self.params.hubert_base
            elif self.args.data.encoder == "hubertbase768l12":
                encoderPath = self.params.hubert_base
            elif self.args.data.encoder == "hubertlarge1024l24":
                encoderPath = self.params.hubert_base
            elif self.args.data.encoder == "contentvec":
                encoderPath = self.params.hubert_base
            elif self.args.data.encoder == "contentvec768":
                encoderPath = self.params.hubert_base
            elif self.args.data.encoder == "contentvec768l12":
                encoderPath = self.params.hubert_base
            
            self.units_encoder = Units_Encoder(
                self.args.data.encoder,
                encoderPath,
                self.args.data.encoder_sample_rate,
                self.args.data.encoder_hop_size,
                cnhubertsoft_gate=cnhubertsoft_gate,
                device=self.device)
            self.encoder_type = self.args.data.encoder
            self.encoder_ckpt = self.args.data.encoder_ckpt
   
    def infer(self,
              audio,
              sample_rate,
              spk_id=1,
              threhold=-45,
              pitch_adjust=0,
              use_spk_mix=False,
              spk_mix_dict=None,
              enhancer_adaptive_key='auto',
              pitch_extractor_type='crepe',
              f0_min=50,
              f0_max=1100,
              safe_prefix_pad_length=0,
              diff_acc=None,
              diff_method='ddim',
              k_step=None,
              diff_silence=False,
              audio_alignment=False
              ):
        # print("Infering...")
        # load input
        # audio, sample_rate = librosa.load(input_wav, sr=None, mono=True)
        hop_size = self.args.data.block_size * sample_rate / self.args.data.sampling_rate
        if audio_alignment:
            audio_length = len(audio)
        # safe front silence
        if safe_prefix_pad_length > 0.03:
            silence_front = safe_prefix_pad_length - 0.03
        else:
            silence_front = 0
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

        # extract f0
        pitch_extractor = F0_Extractor(
            pitch_extractor_type,
            sample_rate,
            hop_size,
            float(f0_min),
            float(f0_max))
        f0 = pitch_extractor.extract(audio, uv_interp=True, device=self.device, silence_front=silence_front)
        f0 = torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        f0 = f0 * 2 ** (float(pitch_adjust) / 12)

        # extract volume
        volume_extractor = Volume_Extractor(hop_size)
        volume = volume_extractor.extract(audio)
        mask = (volume > 10 ** (float(threhold) / 20)).astype('float')
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([np.max(mask[n: n + 9]) for n in range(len(mask) - 8)])
        mask = torch.from_numpy(mask).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, self.args.data.block_size).squeeze(-1)
        volume = torch.from_numpy(volume).float().to(self.device).unsqueeze(-1).unsqueeze(0)

        # extract units
        units = self.units_encoder.encode(audio_t, sample_rate, hop_size)

        # spk_id or spk_mix_dict
        spk_id = torch.LongTensor(np.array([[spk_id]])).to(self.device)
        dictionary = None
        if use_spk_mix:
            dictionary = spk_mix_dict

        # forward and return the output
        with torch.no_grad():
            if self.ddsp_model is None:
                output = None
            else:
                output, _, (s_h, s_n) = self.ddsp_model(units, f0, volume, spk_id=spk_id, spk_mix_dict=dictionary)
            output = self.diff_model.infer(output, f0, units, volume, acc=diff_acc, spk_id=spk_id,
                                      k_step=k_step, method=diff_method, silence_front=silence_front, use_silence=diff_silence,
                                      spk_mix_dict=dictionary)
            output *= mask
            output = output.squeeze()
            if audio_alignment:
                output[:audio_length]
            return output, self.args.data.sampling_rate