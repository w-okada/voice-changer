import numpy as np
import time
import os
import torch
import torch.nn.functional
from torchaudio.transforms import Resample
from tqdm import tqdm
from voice_changer.DiffusionSVC.inferencer.diffusion_svc_model.diffusion.unit2mel import load_model_vocoder, load_model_vocoder_from_combo
from voice_changer.DiffusionSVC.inferencer.diffusion_svc_model.tools.slicer import split
from voice_changer.DiffusionSVC.inferencer.diffusion_svc_model.tools.units_index import UnitsIndexer
from voice_changer.DiffusionSVC.inferencer.diffusion_svc_model.tools.tools import F0_Extractor, Volume_Extractor, Units_Encoder, SpeakerEncoder, cross_fade


class DiffusionSVC:
    def __init__(self, device=None):
        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = None
        self.model = None
        self.vocoder = None
        self.args = None
        # 特征提取器
        self.units_encoder = None
        self.f0_extractor = None
        self.f0_model = None
        self.f0_min = None
        self.f0_max = None
        self.volume_extractor = None
        self.speaker_encoder = None
        self.spk_emb_dict = None
        self.resample_dict_16000 = {}
        self.units_indexer = None
        self.naive_model_path = None
        self.naive_model = None
        self.naive_model_args = None
        self.use_combo_model = False

    def load_model(self, model_path, f0_model=None, f0_min=None, f0_max=None):
        self.use_combo_model = True
        self.model_path = model_path
        self.naive_model_path = model_path
        diff_model, diff_args, naive_model, naive_args, vocoder = load_model_vocoder_from_combo(model_path,
                                                                                                device=self.device)
        self.model = diff_model
        self.args = diff_args
        self.naive_model = naive_model
        self.naive_model_args = naive_args
        self.vocoder = vocoder

        self.units_encoder = Units_Encoder(
            self.args.data.encoder,
            self.args.data.encoder_ckpt,
            self.args.data.encoder_sample_rate,
            self.args.data.encoder_hop_size,
            cnhubertsoft_gate=self.args.data.cnhubertsoft_gate,
            device=self.device,
            units_forced_mode=self.args.data.units_forced_mode
        )

        self.volume_extractor = Volume_Extractor(
            hop_size=512,
            block_size=self.args.data.block_size,
            model_sampling_rate=self.args.data.sampling_rate
        )

        self.load_f0_extractor(f0_model=f0_model, f0_min=f0_min, f0_max=f0_max)

        if self.args.model.use_speaker_encoder:
            self.speaker_encoder = SpeakerEncoder(
                self.args.data.speaker_encoder,
                self.args.data.speaker_encoder_config,
                self.args.data.speaker_encoder_ckpt,
                self.args.data.speaker_encoder_sample_rate,
                device=self.device
            )
            path_spk_emb_dict = os.path.join(os.path.split(model_path)[0], 'spk_emb_dict.npy')
            self.set_spk_emb_dict(path_spk_emb_dict)

        self.units_indexer = UnitsIndexer(os.path.split(model_path)[0])

    def load_f0_extractor(self, f0_model, f0_min=None, f0_max=None):
        self.f0_model = f0_model if (f0_model is not None) else self.args.data.f0_extractor
        self.f0_min = f0_min if (f0_min is not None) else self.args.data.f0_min
        self.f0_max = f0_max if (f0_max is not None) else self.args.data.f0_max
        self.f0_model = f0_model
        self.f0_extractor = F0_Extractor(
            f0_extractor=self.f0_model,
            sample_rate=44100,
            hop_size=512,
            f0_min=self.f0_min,
            f0_max=self.f0_max,
            block_size=self.args.data.block_size,
            model_sampling_rate=self.args.data.sampling_rate
        )


    @torch.no_grad()
    def naive_model_call(self, units, f0, volume, spk_id=1, spk_mix_dict=None,
                         aug_shift=0, spk_emb=None):
        # spk_id
        spk_emb_dict = None
        if self.args.model.use_speaker_encoder:  # with speaker encoder
            spk_mix_dict, spk_emb = self.pre_spk_emb(spk_id, spk_mix_dict, len(units), spk_emb)
        # without speaker encoder
        else:
            spk_id = torch.LongTensor(np.array([[int(spk_id)]])).to(self.device)
        aug_shift = torch.from_numpy(np.array([[float(aug_shift)]])).float().to(self.device)
        out_spec = self.naive_model(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict,
                                    aug_shift=aug_shift, infer=True,
                                    spk_emb=spk_emb, spk_emb_dict=spk_emb_dict)
        return out_spec

    def set_spk_emb_dict(self, spk_emb_dict_or_path):  # 从路径加载或直接设置
        if spk_emb_dict_or_path is None:
            return None
        if spk_emb_dict_or_path is dict:
            self.spk_emb_dict = spk_emb_dict_or_path
            print(f" [INFO] Load spk_emb_dict from {spk_emb_dict_or_path}")
        else:
            self.spk_emb_dict = np.load(spk_emb_dict_or_path, allow_pickle=True).item()
            print(f" [INFO] Load spk_emb_dict from {spk_emb_dict_or_path}")

    @torch.no_grad()
    def encode_units(self, audio, sr=44100, padding_mask=None):
        assert self.units_encoder is not None
        hop_size = self.args.data.block_size * sr / self.args.data.sampling_rate
        return self.units_encoder.encode(audio, sr, hop_size, padding_mask=padding_mask)

    @torch.no_grad()
    def extract_f0(self, audio, key=0, sr=44100, silence_front=0):
        assert self.f0_extractor is not None
        f0 = self.f0_extractor.extract(audio, uv_interp=True, device=self.device, silence_front=silence_front, sr=sr)
        f0 = torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        f0 = f0 * 2 ** (float(key) / 12)
        return f0

    @torch.no_grad()
    def extract_volume_and_mask(self, audio, sr=44100, threhold=-60.0):
        assert self.volume_extractor is not None
        volume = self.volume_extractor.extract(audio, sr)
        mask = self.volume_extractor.get_mask_from_volume(volume, threhold=threhold, device=self.device)
        volume = torch.from_numpy(volume).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        return volume, mask

    @torch.no_grad()
    def extract_mel(self, audio, sr=44100):
        assert sr == 441000
        mel = self.vocoder.extract(audio, self.args.data.sampling_rate)
        return mel

    @torch.no_grad()
    def encode_spk(self, audio, sr=44100):
        assert self.speaker_encoder is not None
        return self.speaker_encoder(audio=audio, sample_rate=sr)

    @torch.no_grad()
    def encode_spk_from_path(self, path):  # 从path读取预先提取的声纹(必须是.npy文件), 或从声音文件提取声纹(此时可以是文件或目录)
        if path is None:
            return None
        assert self.speaker_encoder is not None
        if (('122333444455555' + path)[-4:] == '.npy') and os.path.isfile(path):
            spk_emb = np.load(path)
        else:
            if os.path.isfile(path):
                path_list = [path]
            else:
                path_list = os.listdir(path)
                for _index in range(len(path_list)):
                    path_list[_index] = os.path.join(path, path_list[_index])
            spk_emb = self.speaker_encoder.mean_spk_emb_from_path_list(path_list)
        return spk_emb

    def pre_spk_emb(self, spk_id, spk_mix_dict, units_len, spk_emb):
        spk_emb_dict = self.spk_emb_dict
        if (spk_mix_dict is not None) or (spk_emb is None):
            assert spk_emb_dict is not None
        if spk_emb is None:
            spk_emb = spk_emb_dict[str(spk_id)]
        # pad and to device
        spk_emb = np.tile(spk_emb, (units_len, 1))
        spk_emb = torch.from_numpy(spk_emb).float().to(self.device)
        return spk_mix_dict, spk_emb

    @torch.no_grad()
    def mel2wav(self, mel, f0, start_frame=0):
        if start_frame == 0:
            return self.vocoder.infer(mel, f0)
        else:  # for realtime speedup
            mel = mel[:, start_frame:, :]
            f0 = f0[:, start_frame:, :]
            out_wav = self.vocoder.infer(mel, f0)
            return torch.nn.functional.pad(out_wav, (start_frame * self.vocoder.vocoder_hop_size, 0))

    @torch.no_grad()  # 最基本推理代码,将输入标准化为tensor,只与mel打交道
    def __call__(self, units, f0, volume, spk_id=1, spk_mix_dict=None, aug_shift=0,
                 gt_spec=None, infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True,
                 spk_emb=None):

        if self.args.model.k_step_max is not None:
            if k_step is None:
                raise ValueError("k_step must not None when Shallow Diffusion Model inferring")
            if k_step > int(self.args.model.k_step_max):
                raise ValueError(f"k_step must <= k_step_max of Shallow Diffusion Model")
            if gt_spec is None:
                raise ValueError("gt_spec must not None when Shallow Diffusion Model inferring, gt_spec can from "
                                 "input mel or output of naive model")
            print(f' [INFO] k_step_max is {self.args.model.k_step_max}.')

        aug_shift = torch.from_numpy(np.array([[float(aug_shift)]])).float().to(self.device)

        # spk_id
        spk_emb_dict = None
        if self.args.model.use_speaker_encoder:  # with speaker encoder
            spk_mix_dict, spk_emb = self.pre_spk_emb(spk_id, spk_mix_dict, len(units), spk_emb)
        # without speaker encoder
        else:
            spk_id = torch.LongTensor(np.array([[int(spk_id)]])).to(self.device)

        if k_step is not None:
            print(f' [INFO] get k_step, do shallow diffusion {k_step} steps')
        else:
            print(f' [INFO] Do full 1000 steps depth diffusion {k_step}')
        print(f" [INFO] method:{method}; infer_speedup:{infer_speedup}")
        return self.model(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift,
                          gt_spec=gt_spec, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step,
                          use_tqdm=use_tqdm, spk_emb=spk_emb, spk_emb_dict=spk_emb_dict)


    @torch.no_grad()  # 为实时优化的推理代码，可以切除pad省算力
    def infer_from_audio_for_realtime(self, audio, sr, key, spk_id=1, spk_mix_dict=None, aug_shift=0,
                                      infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True,
                                      spk_emb=None, silence_front=0, diff_jump_silence_front=False, threhold=-60,
                                      index_ratio=0, use_hubert_mask=False):

        start_frame = int(silence_front * self.vocoder.vocoder_sample_rate / self.vocoder.vocoder_hop_size)
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

        if self.naive_model is None:
            print(" [INFO] No combo_model or naive_model, diffusion without shallow-model.")
        else:
            assert k_step is not None
            print(" [INFO] Shallow Diffusion mode!")

        key_str = str(sr)
        if key_str not in self.resample_dict_16000:
            self.resample_dict_16000[key_str] = Resample(sr, 16000, lowpass_filter_width=128).to(self.device)
        if int(sr) != 16000:
            audio_t_16k = self.resample_dict_16000[key_str](audio_t)
        else:
            audio_t_16k = audio_t

        volume, mask = self.extract_volume_and_mask(audio, sr, threhold=float(threhold))
        if use_hubert_mask:
            mask16k = mask.clone().unsqueeze(0).unsqueeze(0)
            mask16k = torch.nn.functional.interpolate(mask16k, size=tuple(audio_t_16k.shape), mode='nearest')
            mask16k = ~(mask16k.squeeze(0).squeeze(0).bool())
        else:
            mask16k = None
        units = self.encode_units(audio_t_16k, sr=16000, padding_mask=mask16k)
        if index_ratio > 0:
            units = self.units_indexer(units_t=units, spk_id=spk_id, ratio=index_ratio)
        f0 = self.extract_f0(audio, key=key, sr=sr, silence_front=silence_front)

        if diff_jump_silence_front:
            audio_t = audio_t[:, start_frame * self.vocoder.vocoder_hop_size:]
            f0 = f0[:, start_frame:, :]
            units = units[:, start_frame:, :]
            volume = volume[:, start_frame:, :]

        if k_step is not None:
            k_step = int(k_step)
            if (k_step >= 1000) or (k_step <= 0):
                k_step = 300
                print(f" [WARN] k_step must < 1000 and > 0, now set to {k_step}")
            if self.args.model.k_step_max is not None:
                k_step_max = int(self.args.model.k_step_max)
                if k_step > k_step_max:
                    print(f" [WARN] k_step must <= k_step_max={k_step_max}, not k_step set to{k_step_max}.")
                    k_step = k_step_max
            if int(k_step/infer_speedup) < 3:
                infer_speedup = int(k_step/4)
                print(f" [WARN] diffusion step must > 4 (3 when qndm), not set to{infer_speedup}")
            if self.naive_model is not None:
                gt_spec = self.naive_model_call(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict,
                                                aug_shift=aug_shift, spk_emb=spk_emb)
            else:
                gt_spec = self.vocoder.extract(audio_t, self.args.data.sampling_rate)
                gt_spec = torch.cat((gt_spec, gt_spec[:, -1:, :]), 1)

        else:
            gt_spec = None

        out_mel = self.__call__(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift,
                                gt_spec=gt_spec, infer_speedup=infer_speedup, method=method, k_step=k_step,
                                use_tqdm=use_tqdm, spk_emb=spk_emb)

        if diff_jump_silence_front:
            out_wav = self.mel2wav(out_mel, f0)
        else:
            out_wav = self.mel2wav(out_mel, f0, start_frame=start_frame)
            out_wav *= mask
        return out_wav.squeeze(), self.args.data.sampling_rate
