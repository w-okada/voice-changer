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


class SvcDDSP:
    def __init__(self) -> None:
        self.model = None
        self.units_encoder = None
        self.encoder_type = None
        self.encoder_ckpt = None
        self.enhancer = None
        self.enhancer_type = None
        self.enhancer_ckpt = None

    def setVCParams(self, params: VoiceChangerParams):
        self.params = params

    def update_model(self, model_path: str, device: torch.device):
        self.device = device

        # load ddsp model
        if self.model is None or self.model_path != model_path:
            self.model, self.args = load_model(model_path, device=self.device)
            self.model_path = model_path

            print("ARGS:", self.args)

            # load units encoder
            if self.units_encoder is None or self.args.data.encoder != self.encoder_type or self.args.data.encoder_ckpt != self.encoder_ckpt:
                if self.args.data.encoder == "cnhubertsoftfish":
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
                    device=self.device,
                )
                self.encoder_type = self.args.data.encoder
                self.encoder_ckpt = encoderPath

        # load enhancer
        if self.enhancer is None or self.args.enhancer.type != self.enhancer_type or self.args.enhancer.ckpt != self.enhancer_ckpt:
            enhancerPath = self.params.nsf_hifigan
            self.enhancer = Enhancer(self.args.enhancer.type, enhancerPath, device=self.device)
            self.enhancer_type = self.args.enhancer.type
            self.enhancer_ckpt = enhancerPath

    def infer(
        self,
        audio,
        sample_rate,
        spk_id=1,
        threhold=-45,
        pitch_adjust=0,
        use_spk_mix=False,
        spk_mix_dict=None,
        use_enhancer=True,
        enhancer_adaptive_key="auto",
        pitch_extractor_type="crepe",
        f0_min=50,
        f0_max=1100,
        safe_prefix_pad_length=0,
        diff_model=None,
        diff_acc=None,
        diff_spk_id=None,
        diff_use=False,
        # diff_use_dpm=False,
        method="pndm",
        k_step=None,
        diff_silence=False,
        audio_alignment=False,
    ):
        # print("Infering...")
        # print("audio", audio)
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
        print("pitch_extractor_type", pitch_extractor_type)
        pitch_extractor = F0_Extractor(pitch_extractor_type, sample_rate, hop_size, float(f0_min), float(f0_max))
        f0 = pitch_extractor.extract(audio, uv_interp=True, device=self.device, silence_front=silence_front)
        f0 = torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        f0 = f0 * 2 ** (float(pitch_adjust) / 12)

        # extract volume
        volume_extractor = Volume_Extractor(hop_size)
        volume = volume_extractor.extract(audio)
        mask = (volume > 10 ** (float(threhold) / 20)).astype("float")
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([np.max(mask[n : n + 9]) for n in range(len(mask) - 8)])  # type: ignore
        mask = torch.from_numpy(mask).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, self.args.data.block_size).squeeze(-1)
        volume = torch.from_numpy(volume).float().to(self.device).unsqueeze(-1).unsqueeze(0)

        # extract units
        units = self.units_encoder.encode(audio_t, sample_rate, hop_size)

        # spk_id or spk_mix_dict
        spk_id = torch.LongTensor(np.array([[spk_id]])).to(self.device)
        diff_spk_id = torch.LongTensor(np.array([[diff_spk_id]])).to(self.device)
        dictionary = None

        if use_spk_mix:
            dictionary = spk_mix_dict

            # forward and return the output
        with torch.no_grad():
            output, _, (s_h, s_n) = self.model(units, f0, volume, spk_id=spk_id, spk_mix_dict=dictionary)

            if diff_use and diff_model is not None:
                output = diff_model.infer(
                    output,
                    f0,
                    units,
                    volume,
                    acc=diff_acc,
                    spk_id=diff_spk_id,
                    k_step=k_step,
                    # use_dpm=diff_use_dpm,
                    method=method,
                    silence_front=silence_front,
                    use_silence=diff_silence,
                    spk_mix_dict=dictionary,
                )
            output *= mask
            if use_enhancer and not diff_use:
                output, output_sample_rate = self.enhancer.enhance(
                    output,
                    self.args.data.sampling_rate,
                    f0,
                    self.args.data.block_size,
                    adaptive_key=enhancer_adaptive_key,
                    silence_front=silence_front,
                )
            else:
                output_sample_rate = self.args.data.sampling_rate

            output = output.squeeze()
            if audio_alignment:
                output[:audio_length]
            return output, output_sample_rate
