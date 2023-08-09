import numpy as np
import torch
from voice_changer.DiffusionSVC.inferencer.Inferencer import Inferencer
from voice_changer.DiffusionSVC.inferencer.diffusion_svc_model.diffusion.naive.naive import Unit2MelNaive
from voice_changer.DiffusionSVC.inferencer.diffusion_svc_model.diffusion.unit2mel import Unit2Mel, load_model_vocoder_from_combo
from voice_changer.DiffusionSVC.inferencer.diffusion_svc_model.diffusion.vocoder import Vocoder
from voice_changer.DiffusionSVC.inferencer.onnx.VocoderOnnx import VocoderOnnx

from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.utils.Timer import Timer


class DiffusionSVCInferencer(Inferencer):
    def __init__(self, vocoder_torch_path, vocoder_onnx_path):
        self.diff_model: Unit2Mel | None = None
        self.naive_model: Unit2MelNaive | None = None
        self.vocoder: Vocoder | None = None
        self.vocoder_onnx_path = vocoder_onnx_path
        self.vocoder_torch_path = vocoder_torch_path
        self.vocoder_onnx = None

    def loadModel(self, file: str, gpu: int):
        self.setProps("DiffusionSVCCombo", file, True, gpu)
        self.dev = DeviceManager.get_instance().getDevice(gpu)

        diff_model, diff_args, naive_model, naive_args = load_model_vocoder_from_combo(file, device=self.dev)
        # vocoder
        try:  # try onnx
            self.vocoder_onnx = VocoderOnnx()
            self.vocoder_onnx.initialize(self.vocoder_onnx_path, gpu)
            print("[Voice Changer] load onnx nsf-hifigan")
            vocoder = None
        except Exception as e:  # noqa
            print("[Voice Changer] load torch nsf-hifigan")
            vocoder = Vocoder("nsf-hifigan", self.vocoder_torch_path, device=self.dev)
            self.vocoder_onnx = None

        self.diff_model = diff_model
        self.naive_model = naive_model
        self.vocoder = vocoder
        self.diff_args = diff_args
        self.naive_args = naive_args

        return self

    def getConfig(self) -> tuple[int, int]:
        model_sampling_rate = int(self.diff_args.data.sampling_rate)
        model_block_size = int(self.diff_args.data.block_size)
        return model_block_size, model_sampling_rate

    @torch.no_grad()  # 最基本推理代码,将输入标准化为tensor,只与mel打交道
    def __call__(self, units, f0, volume, spk_id=1, spk_mix_dict=None, aug_shift=0,
                 gt_spec=None, infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True,
                 spk_emb=None):

        if self.diff_args.model.k_step_max is not None:
            if k_step is None:
                raise ValueError("k_step must not None when Shallow Diffusion Model inferring")
            if k_step > int(self.diff_args.model.k_step_max):
                raise ValueError("k_step must <= k_step_max of Shallow Diffusion Model")
            if gt_spec is None:
                raise ValueError("gt_spec must not None when Shallow Diffusion Model inferring, gt_spec can from "
                                 "input mel or output of naive model")

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
    def naive_model_call(self, units, f0, volume, spk_id=1, spk_mix_dict=None,
                         aug_shift=0, spk_emb=None):
        # spk_id
        spk_emb_dict = None
        if self.diff_args.model.use_speaker_encoder:  # with speaker encoder
            spk_mix_dict, spk_emb = self.pre_spk_emb(spk_id, spk_mix_dict, len(units), spk_emb)
        # without speaker encoder
        else:
            spk_id = torch.LongTensor(np.array([[int(spk_id)]])).to(self.dev)
        aug_shift = torch.from_numpy(np.array([[float(aug_shift)]])).float().to(self.dev)
        out_spec = self.naive_model(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict,
                                    aug_shift=aug_shift, infer=True,
                                    spk_emb=spk_emb, spk_emb_dict=spk_emb_dict)
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
        audio_t: torch.Tensor,
        feats: torch.Tensor,
        pitch: torch.Tensor,
        volume: torch.Tensor,
        mask: torch.Tensor,
        sid: torch.Tensor,
        k_step: int,
        infer_speedup: int,
        silence_front: float,
        skip_diffusion: bool = True,
    ) -> torch.Tensor:
        with Timer("pre-process", False) as t:
            gt_spec = self.naive_model_call(feats, pitch, volume, spk_id=sid, spk_mix_dict=None, aug_shift=0, spk_emb=None)

        # print("[    ----Timer::1: ]", t.secs)

        with Timer("pre-process", False) as t:
            if skip_diffusion == 0:
                out_mel = self.__call__(feats, pitch, volume, spk_id=sid, spk_mix_dict=None, aug_shift=0, gt_spec=gt_spec, infer_speedup=infer_speedup, method='dpm-solver', k_step=k_step, use_tqdm=False, spk_emb=None)
                gt_spec = out_mel
        # print("[    ----Timer::2: ]", t.secs)


        with Timer("pre-process", False) as t:  # NOQA
            if self.vocoder_onnx is None:
                start_frame = int(silence_front * self.vocoder.vocoder_sample_rate / self.vocoder.vocoder_hop_size)
                out_wav = self.mel2wav(gt_spec, pitch, start_frame=start_frame)
                out_wav *= mask
            else:
                out_wav = self.vocoder_onnx.infer(gt_spec, pitch, silence_front, mask)
        # print("[    ----Timer::3: ]", t.secs)

        return out_wav.squeeze()
