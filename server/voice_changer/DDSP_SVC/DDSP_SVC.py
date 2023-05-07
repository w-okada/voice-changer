import json
import sys
import os
from dataclasses import asdict
import numpy as np
import torch
from torchaudio.transforms import Resample
from torch.nn import functional as F

if sys.platform.startswith("darwin"):
    baseDir = [x for x in sys.path if x.endswith("Contents/MacOS")]
    if len(baseDir) != 1:
        print("baseDir should be only one ", baseDir)
        sys.exit()
    modulePath = os.path.join(baseDir[0], "DDSP-SVC")
    sys.path.append(modulePath)
else:
    sys.path.append("DDSP-SVC")

import ddsp.vocoder as vo  # type:ignore
from ddsp.core import upsample  # type:ignore
from enhancer import Enhancer  # type:ignore
from diffusion.infer_gt_mel import DiffGtMel  # type: ignore

from voice_changer.utils.VoiceChangerModel import AudioInOut
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
from voice_changer.utils.LoadModelParams import LoadModelParams
from voice_changer.DDSP_SVC.DDSP_SVCSetting import DDSP_SVCSettings
from voice_changer.RVC.embedder.EmbedderManager import EmbedderManager
from Exceptions import NoModeLoadedException
from voice_changer.DDSP_SVC.SvcDDSP import SvcDDSP


providers = [
    "OpenVINOExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]


def phase_vocoder(a, b, fade_out, fade_in):
    fa = torch.fft.rfft(a)
    fb = torch.fft.rfft(b)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
    t = torch.arange(n).unsqueeze(-1).to(a) / n
    result = (
        a * (fade_out**2)
        + b * (fade_in**2)
        + torch.sum(absab * torch.cos(w * t + phia), -1) * fade_out * fade_in / n
    )
    return result


class DDSP_SVC:
    initialLoad: bool = True
    settings: DDSP_SVCSettings = DDSP_SVCSettings()
    diff_model: DiffGtMel = DiffGtMel()
    svc_model: SvcDDSP = SvcDDSP()
    # diff_model: DiffGtMel = DiffGtMel()

    audio_buffer: AudioInOut | None = None
    prevVol: float = 0
    # resample_kernel = {}

    def __init__(self, params: VoiceChangerParams):
        self.gpu_num = torch.cuda.device_count()
        self.params = params
        self.svc_model.setVCParams(params)
        EmbedderManager.initialize(params)
        print("DDSP-SVC initialization:", params)

    # def useDevice(self):
    #     if self.settings.gpu >= 0 and torch.cuda.is_available():
    #         return torch.device("cuda", index=self.settings.gpu)
    #     else:
    #         return torch.device("cpu")

    def loadModel(self, props: LoadModelParams):
        # target_slot_idx = props.slot
        self.device = torch.device("cuda", index=0)
        params = props.params

        modelFile = params["files"]["ddspSvcModel"]
        diffusionFile = params["files"]["ddspSvcDiffusion"]
        self.svc_model.update_model(modelFile)

        print("diffusion file", diffusionFile)
        self.diff_model.flush_model(diffusionFile, ddsp_config=self.svc_model.args)

        print("params:", params)
        # print("params_arg:", self.args)

        # self.settings.pyTorchModelFile = props.files.pyTorchModelFilename
        # # model
        # model, args = vo.load_model(
        #     self.settings.pyTorchModelFile, device=self.useDevice()
        # )
        # self.model = model
        # self.args = args
        # self.sampling_rate = args.data.sampling_rate
        # self.hop_size = int(
        #     self.args.data.block_size
        #     * self.sampling_rate
        #     / self.args.data.sampling_rate
        # )

        # # hubert
        # self.vec_path = self.params.hubert_soft
        # self.encoder = vo.Units_Encoder(
        #     self.args.data.encoder,
        #     self.vec_path,
        #     self.args.data.encoder_sample_rate,
        #     self.args.data.encoder_hop_size,
        #     device=self.useDevice(),
        # )

        # # f0dec
        # self.f0_detector = vo.F0_Extractor(
        #     # "crepe",
        #     self.settings.f0Detector,
        #     self.sampling_rate,
        #     self.hop_size,
        #     float(50),
        #     float(1100),
        # )

        # self.volume_extractor = vo.Volume_Extractor(self.hop_size)
        # self.enhancer_path = self.params.nsf_hifigan
        # self.enhancer = Enhancer(
        #     self.args.enhancer.type, self.enhancer_path, device=self.useDevice()
        # )
        return self.get_info()

    def update_settings(self, key: str, val: int | float | str):
        #     if key == "onnxExecutionProvider" and self.onnx_session is not None:
        #         if val == "CUDAExecutionProvider":
        #             if self.settings.gpu < 0 or self.settings.gpu >= self.gpu_num:
        #                 self.settings.gpu = 0
        #             provider_options = [{"device_id": self.settings.gpu}]
        #             self.onnx_session.set_providers(
        #                 providers=[val], provider_options=provider_options
        #             )
        #         else:
        #             self.onnx_session.set_providers(providers=[val])
        #     elif key in self.settings.intData:
        #         val = int(val)
        #         setattr(self.settings, key, val)
        #         if (
        #             key == "gpu"
        #             and val >= 0
        #             and val < self.gpu_num
        #             and self.onnx_session is not None
        #         ):
        #             providers = self.onnx_session.get_providers()
        #             print("Providers:", providers)
        #             if "CUDAExecutionProvider" in providers:
        #                 provider_options = [{"device_id": self.settings.gpu}]
        #                 self.onnx_session.set_providers(
        #                     providers=["CUDAExecutionProvider"],
        #                     provider_options=provider_options,
        #                 )
        #         if key == "gpu" and len(self.settings.pyTorchModelFile) > 0:
        #             model, _args = vo.load_model(
        #                 self.settings.pyTorchModelFile, device=self.useDevice()
        #             )
        #             self.model = model
        #             self.enhancer = Enhancer(
        #                 self.args.enhancer.type, self.enhancer_path, device=self.useDevice()
        #             )
        #             self.encoder = vo.Units_Encoder(
        #                 self.args.data.encoder,
        #                 self.vec_path,
        #                 self.args.data.encoder_sample_rate,
        #                 self.args.data.encoder_hop_size,
        #                 device=self.useDevice(),
        #             )

        #     elif key in self.settings.floatData:
        #         setattr(self.settings, key, float(val))
        #     elif key in self.settings.strData:
        #         setattr(self.settings, key, str(val))
        #         if key == "f0Detector":
        #             print("f0Detector update", val)
        #             # if val == "dio":
        #             #     val = "parselmouth"

        #             if hasattr(self, "sampling_rate") is False:
        #                 self.sampling_rate = 44100
        #                 self.hop_size = 512

        #             self.f0_detector = vo.F0_Extractor(
        #                 val, self.sampling_rate, self.hop_size, float(50), float(1100)
        #             )
        #     else:
        #         return False

        return True

    def get_info(self):
        #     data = asdict(self.settings)

        #     data["onnxExecutionProviders"] = (
        #         self.onnx_session.get_providers() if self.onnx_session is not None else []
        #     )
        #     files = ["configFile", "pyTorchModelFile", "onnxModelFile"]
        #     for f in files:
        #         if data[f] is not None and os.path.exists(data[f]):
        #             data[f] = os.path.basename(data[f])
        #         else:
        #             data[f] = ""

        data = {}
        return data

    def get_processing_sampling_rate(self):
        return self.svc_model.args.data.sampling_rate

    def generate_input(
        self,
        newData: AudioInOut,
        inputSize: int,
        crossfadeSize: int,
        solaSearchFrame: int = 0,
    ):
        newData = newData.astype(np.float32) / 32768.0
        # newData = newData.astype(np.float32)

        if self.audio_buffer is not None:
            self.audio_buffer = np.concatenate(
                [self.audio_buffer, newData], 0
            )  # 過去のデータに連結
        else:
            self.audio_buffer = newData

        convertSize = (
            inputSize + crossfadeSize + solaSearchFrame + self.settings.extraConvertSize
        )

        # if convertSize % self.hop_size != 0:  # モデルの出力のホップサイズで切り捨てが発生するので補う。
        #     convertSize = convertSize + (self.hop_size - (convertSize % self.hop_size))

        convertOffset = -1 * convertSize
        self.audio_buffer = self.audio_buffer[convertOffset:]  # 変換対象の部分だけ抽出

        # # f0
        # f0 = self.f0_detector.extract(
        #     self.audio_buffer * 32768.0,
        #     uv_interp=True,
        #     silence_front=self.settings.extraConvertSize / self.sampling_rate,
        # )
        # f0 = torch.from_numpy(f0).float().unsqueeze(-1).unsqueeze(0)
        # f0 = f0 * 2 ** (float(self.settings.tran) / 12)

        # # volume, mask
        # volume = self.volume_extractor.extract(self.audio_buffer)
        # mask = (volume > 10 ** (float(-60) / 20)).astype("float")
        # mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        # mask = np.array(
        #     [np.max(mask[n : n + 9]) for n in range(len(mask) - 8)]  # noqa: E203
        # )
        # mask = torch.from_numpy(mask).float().unsqueeze(-1).unsqueeze(0)
        # mask = upsample(mask, self.args.data.block_size).squeeze(-1)
        # volume = torch.from_numpy(volume).float().unsqueeze(-1).unsqueeze(0)

        # # embed
        # audio = (
        #     torch.from_numpy(self.audio_buffer)
        #     .float()
        #     .to(self.useDevice())
        #     .unsqueeze(0)
        # )
        # seg_units = self.encoder.encode(audio, self.sampling_rate, self.hop_size)

        # cropOffset = -1 * (inputSize + crossfadeSize)
        # cropEnd = -1 * (crossfadeSize)
        # crop = self.audio_buffer[cropOffset:cropEnd]

        # rms = np.sqrt(np.square(crop).mean(axis=0))
        # vol = max(rms, self.prevVol * 0.0)
        # self.prevVol = vol

        return (self.audio_buffer, inputSize, crossfadeSize, solaSearchFrame)

    # def _onnx_inference(self, data):
    #     if hasattr(self, "onnx_session") is False or self.onnx_session is None:
    #         print("[Voice Changer] No onnx session.")
    #         raise NoModeLoadedException("ONNX")

    #     raise NoModeLoadedException("ONNX")

    def _pyTorch_inference(self, data):
        # if hasattr(self, "model") is False or self.model is None:
        #     print("[Voice Changer] No pyTorch session.")
        #     raise NoModeLoadedException("pytorch")

        input_wav = data[0]
        # inputSize = data[1]
        # crossfadeSize = data[2]
        # solaSearchFrame = data[3]
        # last_delay_frame = int(0.02 * self.svc_model.args.data.sampling_rate)

        # fade_in_window = (
        #     torch.sin(
        #         np.pi * torch.arange(0, 1, 1 / crossfadeSize, device=self.device) / 2
        #     )
        #     ** 2
        # )
        # fade_out_window = 1 - fade_in_window

        _audio, _model_sr = self.svc_model.infer(
            input_wav,
            44100,
            spk_id=1,
            threhold=-45,
            pitch_adjust=10,
            use_spk_mix=False,
            spk_mix_dict=None,
            use_enhancer=False,
            pitch_extractor_type="harvest",
            f0_min=50,
            f0_max=1100,
            safe_prefix_pad_length=0,  # TBD なにこれ？
            diff_model=self.diff_model,
            diff_acc=20,  # TBD なにこれ？
            diff_spk_id=1,
            diff_use=True,
            diff_use_dpm=False,  # TBD なにこれ？
            k_step=120,  # TBD なにこれ？
            diff_silence=False,  # TBD なにこれ？
        )

        print(" _model_sr", _model_sr)
        print("_audio", _audio.shape)
        print("_audio", _audio)
        return _audio.cpu().numpy() * 32768.0

        # if _model_sr != self.svc_model.args.data.sampling_rate:
        #     key_str = str(_model_sr) + "_" + str(self.svc_model.args.data.sampling_rate)
        #     if key_str not in self.resample_kernel:
        #         self.resample_kernel[key_str] = Resample(
        #             _model_sr,
        #             self.svc_model.args.data.sampling_rate,
        #             lowpass_filter_width=128,
        #         ).to(self.device)
        #     _audio = self.resample_kernel[key_str](_audio)
        # temp_wav = _audio[
        #     -inputSize
        #     - crossfadeSize
        #     - solaSearchFrame
        #     - last_delay_frame : -last_delay_frame
        # ]

        # # sola shift
        # conv_input = temp_wav[None, None, : crossfadeSize + solaSearchFrame]
        # cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        # cor_den = torch.sqrt(
        #     F.conv1d(
        #         conv_input**2,
        #         torch.ones(1, 1, crossfadeSize, device=self.device),
        #     )
        #     + 1e-8
        # )
        # sola_shift = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        # temp_wav = temp_wav[sola_shift : sola_shift + inputSize + crossfadeSize]
        # print("sola_shift: " + str(int(sola_shift)))

        # # phase vocoder
        # # if self.config.use_phase_vocoder:
        # if False:
        #     temp_wav[:crossfadeSize] = phase_vocoder(
        #         self.sola_buffer,
        #         temp_wav[:crossfadeSize],
        #         fade_out_window,
        #         fade_in_window,
        #     )
        # else:
        #     temp_wav[:crossfadeSize] *= fade_in_window
        #     temp_wav[:crossfadeSize] += self.sola_buffer * fade_out_window

        # self.sola_buffer = temp_wav[-crossfadeSize:]

        # result = temp_wav[:-crossfadeSize, None].repeat(1, 2).cpu().numpy()

        ###########################################
        # c = data[0].to(self.useDevice())
        # f0 = data[1].to(self.useDevice())
        # volume = data[2].to(self.useDevice())
        # mask = data[3].to(self.useDevice())

        # # convertSize = data[4]
        # # vol = data[5]
        # # if vol < self.settings.silentThreshold:
        # #     print("threshold")
        # #     return np.zeros(convertSize).astype(np.int16)

        # with torch.no_grad():
        #     spk_id = torch.LongTensor(np.array([[self.settings.dstId]])).to(
        #         self.useDevice()
        #     )
        #     seg_output, _, (s_h, s_n) = self.model(
        #         c, f0, volume, spk_id=spk_id, spk_mix_dict=None
        #     )
        #     seg_output *= mask

        #     if self.settings.enableEnhancer:
        #         seg_output, output_sample_rate = self.enhancer.enhance(
        #             seg_output,
        #             self.args.data.sampling_rate,
        #             f0,
        #             self.args.data.block_size,
        #             # adaptive_key=float(self.settings.enhancerTune),
        #             adaptive_key="auto",
        #             silence_front=self.settings.extraConvertSize / self.sampling_rate,
        #         )

        #     result = seg_output.squeeze().cpu().numpy() * 32768.0

    #        return np.array(result).astype(np.int16)

    def inference(self, data):
        if self.settings.framework == "ONNX":
            audio = self._onnx_inference(data)
        else:
            audio = self._pyTorch_inference(data)
        return audio

    # def destroy(self):
    #     del self.net_g
    #     del self.onnx_session

    def __del__(self):
        del self.net_g
        del self.onnx_session

        remove_path = os.path.join("DDSP-SVC")
        sys.path = [x for x in sys.path if x.endswith(remove_path) is False]

        for key in list(sys.modules):
            val = sys.modules.get(key)
            try:
                file_path = val.__file__
                if file_path.find("DDSP-SVC" + os.path.sep) >= 0:
                    print("remove", key, file_path)
                    sys.modules.pop(key)
            except:  # type:ignore
                pass
