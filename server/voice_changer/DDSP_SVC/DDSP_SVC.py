import sys
import os
from dataclasses import asdict
import numpy as np
import torch
from data.ModelSlot import DDSPSVCModelSlot
from voice_changer.DDSP_SVC.ModelSlot import ModelSlot

from voice_changer.DDSP_SVC.deviceManager.DeviceManager import DeviceManager

if sys.platform.startswith("darwin"):
    baseDir = [x for x in sys.path if x.endswith("Contents/MacOS")]
    if len(baseDir) != 1:
        print("baseDir should be only one ", baseDir)
        sys.exit()
    modulePath = os.path.join(baseDir[0], "DDSP-SVC")
    sys.path.append(modulePath)
else:
    sys.path.append("DDSP-SVC")

from diffusion.infer_gt_mel import DiffGtMel  # type: ignore

from voice_changer.utils.VoiceChangerModel import AudioInOut
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
from voice_changer.utils.LoadModelParams import LoadModelParams, LoadModelParams2
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
    result = a * (fade_out**2) + b * (fade_in**2) + torch.sum(absab * torch.cos(w * t + phia), -1) * fade_out * fade_in / n
    return result


class DDSP_SVC:
    initialLoad: bool = True
    settings: DDSP_SVCSettings = DDSP_SVCSettings()
    diff_model: DiffGtMel = DiffGtMel()
    svc_model: SvcDDSP = SvcDDSP()

    deviceManager = DeviceManager.get_instance()
    # diff_model: DiffGtMel = DiffGtMel()

    audio_buffer: AudioInOut | None = None
    prevVol: float = 0
    # resample_kernel = {}

    def __init__(self, params: VoiceChangerParams):
        self.gpu_num = torch.cuda.device_count()
        self.params = params
        self.svc_model.setVCParams(params)
        EmbedderManager.initialize(params)
        print("[Voice Changer] DDSP-SVC initialization:", params)

    def loadModel(self, props: LoadModelParams):
        target_slot_idx = props.slot
        params = props.params

        modelFile = params["files"]["ddspSvcModel"]
        diffusionFile = params["files"]["ddspSvcDiffusion"]
        modelSlot = ModelSlot(
            modelFile=modelFile,
            diffusionFile=diffusionFile,
            defaultTrans=params["trans"] if "trans" in params else 0,
        )
        self.settings.modelSlots[target_slot_idx] = modelSlot

        # 初回のみロード
        # if self.initialLoad:
        #     self.prepareModel(target_slot_idx)
        #     self.settings.modelSlotIndex = target_slot_idx
        #     self.switchModel()
        #     self.initialLoad = False
        # elif target_slot_idx == self.currentSlot:
        #     self.prepareModel(target_slot_idx)
        self.settings.modelSlotIndex = target_slot_idx
        self.reloadModel()

        print("params:", params)
        return self.get_info()

    def reloadModel(self):
        self.device = self.deviceManager.getDevice(self.settings.gpu)
        modelFile = self.settings.modelSlots[self.settings.modelSlotIndex].modelFile
        diffusionFile = self.settings.modelSlots[self.settings.modelSlotIndex].diffusionFile

        self.svc_model = SvcDDSP()
        self.svc_model.setVCParams(self.params)
        self.svc_model.update_model(modelFile, self.device)
        self.diff_model = DiffGtMel(device=self.device)
        self.diff_model.flush_model(diffusionFile, ddsp_config=self.svc_model.args)

    def update_settings(self, key: str, val: int | float | str):
        if key in self.settings.intData:
            val = int(val)
            setattr(self.settings, key, val)
            if key == "gpu":
                self.reloadModel()
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
            self.audio_buffer = np.concatenate([self.audio_buffer, newData], 0)  # 過去のデータに連結
        else:
            self.audio_buffer = newData

        convertSize = inputSize + crossfadeSize + solaSearchFrame + self.settings.extraConvertSize

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
        # if hasattr(self, "model") is False or self.model is None:
        #     print("[Voice Changer] No pyTorch session.")
        #     raise NoModeLoadedException("pytorch")

        input_wav = data[0]
        _audio, _model_sr = self.svc_model.infer(
            input_wav,
            self.svc_model.args.data.sampling_rate,
            spk_id=self.settings.dstId,
            threhold=self.settings.threshold,
            pitch_adjust=self.settings.tran,
            use_spk_mix=False,
            spk_mix_dict=None,
            use_enhancer=True if self.settings.useEnhancer == 1 else False,
            pitch_extractor_type=self.settings.f0Detector,
            f0_min=50,
            f0_max=1100,
            # safe_prefix_pad_length=0,  # TBD なにこれ？
            safe_prefix_pad_length=self.settings.extraConvertSize / self.svc_model.args.data.sampling_rate,
            diff_model=self.diff_model,
            diff_acc=self.settings.diffAcc,  # TBD なにこれ？
            diff_spk_id=self.settings.diffSpkId,
            diff_use=True if self.settings.useDiff == 1 else False,
            # diff_use_dpm=True if self.settings.useDiffDpm == 1 else False,  # TBD なにこれ？
            method=self.settings.diffMethod,
            k_step=self.settings.kStep,  # TBD なにこれ？
            diff_silence=True if self.settings.useDiffSilence == 1 else False,  # TBD なにこれ？
        )

        return _audio.cpu().numpy() * 32768.0

    def inference(self, data):
        if self.settings.framework == "ONNX":
            audio = self._onnx_inference(data)
        else:
            audio = self._pyTorch_inference(data)
        return audio

    @classmethod
    def loadModel2(cls, props: LoadModelParams2):
        slotInfo: DDSPSVCModelSlot = DDSPSVCModelSlot()
        for file in props.files:
            if file.kind == "ddspSvcModelConfig":
                slotInfo.configFile = file.name
            elif file.kind == "ddspSvcModel":
                slotInfo.modelFile = file.name
            elif file.kind == "ddspSvcDiffusionConfig":
                slotInfo.diffConfigFile = file.name
            elif file.kind == "ddspSvcDiffusion":
                slotInfo.diffModelFile = file.name
        slotInfo.isONNX = slotInfo.modelFile.endswith(".onnx")
        slotInfo.name = os.path.splitext(os.path.basename(slotInfo.modelFile))[0]
        return slotInfo

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
                    # print("remove", key, file_path)
                    sys.modules.pop(key)
            except:  # type:ignore
                pass
