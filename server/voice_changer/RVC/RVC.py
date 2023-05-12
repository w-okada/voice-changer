import sys
import os
import resampy
from dataclasses import asdict
from typing import cast
import numpy as np
import torch


# avoiding parse arg error in RVC
sys.argv = ["MMVCServerSIO.py"]

if sys.platform.startswith("darwin"):
    baseDir = [x for x in sys.path if x.endswith("Contents/MacOS")]
    if len(baseDir) != 1:
        print("baseDir should be only one ", baseDir)
        sys.exit()
    modulePath = os.path.join(baseDir[0], "RVC")
    sys.path.append(modulePath)
else:
    sys.path.append("RVC")

from voice_changer.RVC.modelMerger.MergeModel import merge_model
from voice_changer.RVC.modelMerger.MergeModelRequest import MergeModelRequest
from voice_changer.RVC.ModelSlotGenerator import generateModelSlot
from voice_changer.RVC.RVCSettings import RVCSettings
from voice_changer.RVC.embedder.EmbedderManager import EmbedderManager
from voice_changer.utils.LoadModelParams import LoadModelParams
from voice_changer.utils.VoiceChangerModel import AudioInOut
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
from voice_changer.RVC.onnxExporter.export2onnx import export2onnx
from voice_changer.RVC.pitchExtractor.PitchExtractorManager import PitchExtractorManager
from voice_changer.RVC.pipeline.PipelineGenerator import createPipeline
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.pipeline.Pipeline import Pipeline

from Exceptions import NoModeLoadedException
from const import UPLOAD_DIR


providers = [
    "OpenVINOExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]


class RVC:
    initialLoad: bool = True
    settings: RVCSettings = RVCSettings()

    pipeline: Pipeline | None = None

    deviceManager = DeviceManager.get_instance()

    audio_buffer: AudioInOut | None = None
    prevVol: float = 0
    params: VoiceChangerParams
    currentSlot: int = -1
    needSwitch: bool = False

    def __init__(self, params: VoiceChangerParams):
        self.pitchExtractor = PitchExtractorManager.getPitchExtractor(
            self.settings.f0Detector
        )
        self.params = params
        EmbedderManager.initialize(params)
        print("RVC initialization: ", params)

    def loadModel(self, props: LoadModelParams):
        target_slot_idx = props.slot
        params = props.params

        modelSlot = generateModelSlot(params)
        self.settings.modelSlots[target_slot_idx] = modelSlot
        print(
            f"[Voice Changer] RVC new model is uploaded,{target_slot_idx}",
            asdict(modelSlot),
        )

        # 初回のみロード
        if self.initialLoad:
            self.prepareModel(target_slot_idx)
            self.settings.modelSlotIndex = target_slot_idx
            self.switchModel()
            self.initialLoad = False
        elif target_slot_idx == self.currentSlot:
            self.prepareModel(target_slot_idx)

        return self.get_info()

    def update_settings(self, key: str, val: int | float | str):
        if key in self.settings.intData:
            # 設定前処理
            val = cast(int, val)
            if key == "modelSlotIndex":
                if val < 0:
                    return True
                val = val % 1000  # Quick hack for same slot is selected
                if (
                    self.settings.modelSlots[val].modelFile is None
                    or self.settings.modelSlots[val].modelFile == ""
                ):
                    print("[Voice Changer] slot does not have model.")
                    return True
                self.prepareModel(val)

            # 設定
            setattr(self.settings, key, val)

            if key == "gpu":
                dev = self.deviceManager.getDevice(val)
                half = self.deviceManager.halfPrecisionAvailable(val)

                # half-precisionの使用可否が変わるときは作り直し
                if self.pipeline is not None and self.pipeline.isHalf == half:
                    print(
                        "USE EXSISTING PIPELINE",
                        half,
                    )
                    self.pipeline.setDevice(dev)
                else:
                    print("CHAGE TO NEW PIPELINE", half)
                    self.prepareModel(self.settings.modelSlotIndex)
            if key == "enableDirectML":
                if self.pipeline is not None and val == 0:
                    self.pipeline.setDirectMLEnable(False)
                elif self.pipeline is not None and val == 1:
                    self.pipeline.setDirectMLEnable(True)

        elif key in self.settings.floatData:
            setattr(self.settings, key, float(val))
        elif key in self.settings.strData:
            setattr(self.settings, key, str(val))
            if key == "f0Detector" and self.pipeline is not None:
                pitchExtractor = PitchExtractorManager.getPitchExtractor(
                    self.settings.f0Detector
                )
                self.pipeline.setPitchExtractor(pitchExtractor)
        else:
            return False
        return True

    def prepareModel(self, slot: int):
        if slot < 0:
            return self.get_info()
        modelSlot = self.settings.modelSlots[slot]

        print("[Voice Changer] Prepare Model of slot:", slot)

        # pipelineの生成
        self.next_pipeline = createPipeline(
            modelSlot, self.settings.gpu, self.settings.f0Detector
        )

        # その他の設定
        self.next_trans = modelSlot.defaultTrans
        self.next_samplingRate = modelSlot.samplingRate
        self.next_framework = "ONNX" if modelSlot.isONNX else "PyTorch"
        self.needSwitch = True
        print("[Voice Changer] Prepare done.")
        return self.get_info()

    def switchModel(self):
        print("[Voice Changer] Switching model..")
        self.pipeline = self.next_pipeline
        self.settings.tran = self.next_trans
        self.settings.modelSamplingRate = self.next_samplingRate
        self.settings.framework = self.next_framework

        print(
            "[Voice Changer] Switching model..done",
        )

    def get_info(self):
        data = asdict(self.settings)
        return data

    def get_processing_sampling_rate(self):
        return self.settings.modelSamplingRate

    def generate_input(
        self,
        newData: AudioInOut,
        inputSize: int,
        crossfadeSize: int,
        solaSearchFrame: int = 0,
    ):
        newData = newData.astype(np.float32) / 32768.0

        if self.audio_buffer is not None:
            # 過去のデータに連結
            self.audio_buffer = np.concatenate([self.audio_buffer, newData], 0)
        else:
            self.audio_buffer = newData

        convertSize = (
            inputSize + crossfadeSize + solaSearchFrame + self.settings.extraConvertSize
        )

        if convertSize % 128 != 0:  # モデルの出力のホップサイズで切り捨てが発生するので補う。
            convertSize = convertSize + (128 - (convertSize % 128))

        convertOffset = -1 * convertSize
        self.audio_buffer = self.audio_buffer[convertOffset:]  # 変換対象の部分だけ抽出

        # 出力部分だけ切り出して音量を確認。(TODO:段階的消音にする)
        cropOffset = -1 * (inputSize + crossfadeSize)
        cropEnd = -1 * (crossfadeSize)
        crop = self.audio_buffer[cropOffset:cropEnd]
        rms = np.sqrt(np.square(crop).mean(axis=0))
        vol = max(rms, self.prevVol * 0.0)
        self.prevVol = vol

        return (self.audio_buffer, convertSize, vol)

    def inference(self, data):
        if self.settings.modelSlotIndex < 0:
            print(
                "[Voice Changer] wait for loading model...",
                self.settings.modelSlotIndex,
                self.currentSlot,
            )
            raise NoModeLoadedException("model_common")
        if self.needSwitch:
            print(
                f"[Voice Changer] Switch model {self.currentSlot} -> {self.settings.modelSlotIndex}"
            )
            self.currentSlot = self.settings.modelSlotIndex
            self.switchModel()
            self.needSwitch = False

        half = self.deviceManager.halfPrecisionAvailable(self.settings.gpu)

        audio = data[0]
        convertSize = data[1]
        vol = data[2]

        audio = resampy.resample(audio, self.settings.modelSamplingRate, 16000)

        if vol < self.settings.silentThreshold:
            return np.zeros(convertSize).astype(np.int16)

        repeat = 3 if half else 1
        repeat *= self.settings.rvcQuality  # 0 or 3
        sid = 0
        f0_up_key = self.settings.tran
        index_rate = self.settings.indexRatio
        if_f0 = 1 if self.settings.modelSlots[self.currentSlot].f0 else 0

        embChannels = self.settings.modelSlots[self.currentSlot].embChannels

        audio_out = self.pipeline.exec(
            sid,
            audio,
            f0_up_key,
            index_rate,
            if_f0,
            self.settings.extraConvertSize / self.settings.modelSamplingRate,
            embChannels,
            repeat,
        )

        result = audio_out * np.sqrt(vol)

        return result

    def __del__(self):
        del self.pipeline

        print("---------- REMOVING ---------------")

        remove_path = os.path.join("RVC")
        sys.path = [x for x in sys.path if x.endswith(remove_path) is False]

        for key in list(sys.modules):
            val = sys.modules.get(key)
            try:
                file_path = val.__file__
                if file_path.find("RVC" + os.path.sep) >= 0:
                    print("remove", key, file_path)
                    sys.modules.pop(key)
            except Exception:  # type:ignore
                # print(e)
                pass

    def export2onnx(self):
        modelSlot = self.settings.modelSlots[self.settings.modelSlotIndex]

        if modelSlot.isONNX:
            print("[Voice Changer] export2onnx, No pyTorch filepath.")
            return {"status": "ng", "path": ""}

        output_file_simple = export2onnx(self.settings.gpu, modelSlot)
        return {
            "status": "ok",
            "path": f"/tmp/{output_file_simple}",
            "filename": output_file_simple,
        }

    def merge_models(self, request: str):
        print("[Voice Changer] MergeRequest:", request)
        req: MergeModelRequest = MergeModelRequest.from_json(request)
        merged = merge_model(req)
        targetSlot = 0
        if req.slot < 0:
            targetSlot = len(self.settings.modelSlots) - 1
        else:
            targetSlot = req.slot

        storeDir = os.path.join(UPLOAD_DIR, f"{targetSlot}")
        print("[Voice Changer] store merged model to:", storeDir)
        os.makedirs(storeDir, exist_ok=True)
        storeFile = os.path.join(storeDir, "merged.pth")
        torch.save(merged, storeFile)

        params = {"trans": req.defaultTrans, "files": {"rvcModel": storeFile}}
        props: LoadModelParams = LoadModelParams(
            slot=targetSlot, isHalf=True, params=params
        )
        self.loadModel(props)
        self.prepareModel(targetSlot)
        self.settings.modelSlotIndex = targetSlot
        self.currentSlot = self.settings.modelSlotIndex
