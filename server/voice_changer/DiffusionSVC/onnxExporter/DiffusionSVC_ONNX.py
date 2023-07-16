import numpy as np
import torch
from voice_changer.DiffusionSVC.inferencer.diffusion_svc_model.diffusion.unit2mel import load_model_vocoder_from_combo
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager


class DiffusionSVC_ONNX:
    def __init__(self, file: str, gpu: int):
        self.dev = DeviceManager.get_instance().getDevice(gpu)
        diff_model, diff_args, naive_model, naive_args, vocoder = load_model_vocoder_from_combo(file, device=self.dev)
        self.diff_model = diff_model
        self.naive_model = naive_model
        self.vocoder = vocoder
        self.diff_args = diff_args
        self.naive_args = naive_args

    def forward(self, phone, phone_lengths, sid, max_len=None, convert_length=None):
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec.infer_realtime((z * x_mask)[:, :, :max_len], g=g, convert_length=convert_length)
        return o, x_mask, (z, z_p, m_p, logs_p)


    @torch.no_grad()  # 最基本推理代码,将输入标准化为tensor,只与mel打交道
    def __call__(self, units, f0, volume, spk_id=1, spk_mix_dict=None, aug_shift=0,
                 gt_spec=None, infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True,
                 spk_emb=None):


        aug_shift = torch.from_numpy(np.array([[float(aug_shift)]])).float().to(self.dev)

        # spk_id
        spk_emb_dict = None
        if self.diff_args.model.use_speaker_encoder:  # with speaker encoder
            spk_mix_dict, spk_emb = self.pre_spk_emb(spk_id, spk_mix_dict, len(units), spk_emb)
        # without speaker encoder
        else:
            spk_id = torch.LongTensor(np.array([[int(spk_id)]])).to(self.dev)

        return self.diff_model(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift, gt_spec=gt_spec, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm, spk_emb=spk_emb, spk_emb_dict=spk_emb_dict)

    @torch.no_grad()
    def naive_model_call(self, units, f0, volume, spk_id=1, spk_mix_dict=None,aug_shift=0, spk_emb=None):
        # spk_id
        spk_emb_dict = None
        if self.diff_args.model.use_speaker_encoder:  # with speaker encoder
            spk_mix_dict, spk_emb = self.pre_spk_emb(spk_id, spk_mix_dict, len(units), spk_emb)
        # without speaker encoder
        else:

        return out_spec

    @torch.no_grad()
    def mel2wav(self, mel, f0, start_frame=0):
        if start_frame == 0:
            return self.vocoder.infer(mel, f0)
        else:  # for realtime speedup
            mel = mel[:, start_frame:, :]
            f0 = f0[:, start_frame:, :]
            out_wav = self.vocoder.infer(mel, f0)
            return torch.nn.functional.pad(out_wav, (start_frame * self.vocoder.vocoder_hop_size, 0))

    @torch.no_grad()
    def infer(
        self,
        feats: torch.Tensor,
        pitch: torch.Tensor,
        volume: torch.Tensor,
        mask: torch.Tensor,
        sid: torch.Tensor,
        k_step: int,
        infer_speedup: int,
        silence_front: float,
    ) -> torch.Tensor:
        
        aug_shift = torch.LongTensor([0]).to(feats.device)
        out_spec = self.naive_model(feats, pitch, volume, sid, spk_mix_dict=None,
                                    aug_shift=aug_shift, infer=True,
                                    spk_emb=None, spk_emb_dict=None)


        gt_spec = self.naive_model_call(feats, pitch, volume, spk_id=sid, spk_mix_dict=None, aug_shift=0, spk_emb=None)
        out_mel = self.__call__(feats, pitch, volume, spk_id=sid, spk_mix_dict=None, aug_shift=0, gt_spec=gt_spec, infer_speedup=infer_speedup, method='dpm-solver', k_step=k_step, use_tqdm=False, spk_emb=None)
        start_frame = int(silence_front * self.vocoder.vocoder_sample_rate / self.vocoder.vocoder_hop_size)
        out_wav = self.mel2wav(out_mel, pitch, start_frame=start_frame)

        out_wav *= mask
        return out_wav.squeeze()
