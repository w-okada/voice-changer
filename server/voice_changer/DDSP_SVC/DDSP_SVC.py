import sys
import os
from dataclasses import asdict
import numpy as np
import torch
from data.ModelSlot import DDSPSVCModelSlot

from voice_changer.DDSP_SVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.VoiceChangerParamsManager import VoiceChangerParamsManager

if sys.platform.startswith("darwin"):
    baseDir = [x for x in sys.path if x.endswith("Contents/MacOS")]
    if len(baseDir) != 1:
        print("baseDir should be only one ", baseDir)
        sys.exit()
    modulePath = os.path.join(baseDir[0], "DDSP-SVC")
    sys.path.append(modulePath)
else:
    sys.path.append("DDSP-SVC")

from .models.diffusion.infer_gt_mel import DiffGtMel

from voice_changer.utils.VoiceChangerModel import AudioInOut, VoiceChangerModel
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
from voice_changer.DDSP_SVC.DDSP_SVCSetting import DDSP_SVCSettings
from voice_changer.RVC.embedder.EmbedderManager import EmbedderManager

# from Exceptions import NoModeLoadedException
from voice_changer.DDSP_SVC.SvcDDSP import SvcDDSP


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


class DDSP_SVC(VoiceChangerModel):
    initialLoad: bool = True

    def __init__(self, params: VoiceChangerParams, slotInfo: DDSPSVCModelSlot):
        print("[Voice Changer] [DDSP-SVC] Creating instance ")
        self.voiceChangerType = "DDSP-SVC"
        self.deviceManager = DeviceManager.get_instance()
        self.gpu_num = torch.cuda.device_count()
        self.params = params
        self.settings = DDSP_SVCSettings()
        self.svc_model: SvcDDSP = SvcDDSP()
        # self.diff_model: DiffGtMel = DiffGtMel()

        self.svc_model.setVCParams(params)
        EmbedderManager.initialize(params)

        self.audio_buffer: AudioInOut | None = None
        self.prevVol = 0.0
        self.slotInfo = slotInfo
        self.initialize()

    def initialize(self):
        self.device = self.deviceManager.getDevice(self.settings.gpu)
        vcparams = VoiceChangerParamsManager.get_instance().params
        modelPath = os.path.join(
            vcparams.model_dir,
            str(self.slotInfo.slotIndex),
            "model",
            self.slotInfo.modelFile,
        ) if self.slotInfo.modelFile != 'builtin' else self.slotInfo.modelFile
        diffPath = os.path.join(
            vcparams.model_dir,
            str(self.slotInfo.slotIndex),
            "diff",
            self.slotInfo.diffModelFile,
        )
        
        self.svc_model = SvcDDSP()
        self.svc_model.setVCParams(self.params)
        self.svc_model.update_model(modelPath, diffPath , self.device)
        # self.diff_model = DiffGtMel(device=self.device)
        # self.diff_model.flush_model(diffPath, ddsp_config=self.svc_model.args)

    def update_settings(self, key: str, val: int | float | str):
        if key in self.settings.intData:
            val = int(val)
            setattr(self.settings, key, val)
            if key == "gpu":
                self.initialize()
        elif key in self.settings.floatData:
            setattr(self.settings, key, float(val))
        elif key in self.settings.strData:
            setattr(self.settings, key, str(val))
        else:
            return False
        return True

    def get_info(self):
        data = asdict(self.settings)
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
        return (self.audio_buffer,)

    # def _onnx_inference(self, data):
    #     if hasattr(self, "onnx_session") is False or self.onnx_session is None:
    #         print("[Voice Changer] No onnx session.")
    #         raise NoModeLoadedException("ONNX")

    #     raise NoModeLoadedException("ONNX")

    def _pyTorch_inference(self, data):
        input_wav = data[0]
        _audio, _model_sr = self.svc_model.infer(
            input_wav,
            self.svc_model.args.data.sampling_rate,
            spk_id=self.settings.dstId,
            threhold=self.settings.threshold,
            pitch_adjust=self.settings.tran,
            use_spk_mix=False,
            spk_mix_dict=None,
            # use_enhancer=True if self.settings.useEnhancer == 1 else False,
            pitch_extractor_type=self.settings.f0Detector,
            f0_min=50,
            f0_max=1100,
            # safe_prefix_pad_length=0,  # TBD なにこれ？
            safe_prefix_pad_length=self.settings.extraConvertSize
            / self.svc_model.args.data.sampling_rate,
            # diff_model=self.diff_model,
            diff_acc=self.settings.diffAcc,  # TBD なにこれ？
            # diff_spk_id=self.settings.diffSpkId,
            # diff_use=True if self.settings.useDiff == 1 else False,
            # diff_use_dpm=True if self.settings.useDiffDpm == 1 else False,  # TBD なにこれ？
            diff_method=self.settings.diffMethod,
            k_step=self.settings.kStep,  # TBD なにこれ？
            diff_silence=True
            if self.settings.useDiffSilence == 1
            else False,  # TBD なにこれ？
        )

        return _audio.cpu().numpy() * 32768.0

    def inference(self, data):
        if self.slotInfo.isONNX:
            audio = self._onnx_inference(data)
        else:
            audio = self._pyTorch_inference(data)
        return audio

    def __del__(self):
        remove_path = os.path.join("DDSP-SVC")
        sys.path = [x for x in sys.path if x.endswith(remove_path) is False]

        for key in list(sys.modules):
            val = sys.modules.get(key)
            try:
                file_path = val.__file__
                if file_path.find("DDSP-SVC" + os.path.sep) >= 0:
                    # print("remove", key, file_path)
                    sys.modules.pop(key)
            except:  # type:ignore # noqa
                pass

    def get_model_current(self):
        return []
