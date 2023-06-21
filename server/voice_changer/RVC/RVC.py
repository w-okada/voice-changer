import sys
import os
from dataclasses import asdict
import numpy as np
import torch
import torchaudio
from data.ModelSlot import RVCModelSlot


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


from voice_changer.RVC.RVCSettings import RVCSettings
from voice_changer.RVC.embedder.EmbedderManager import EmbedderManager
from voice_changer.utils.VoiceChangerModel import AudioInOut, VoiceChangerModel
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
from voice_changer.RVC.onnxExporter.export2onnx import export2onnx
from voice_changer.RVC.pitchExtractor.PitchExtractorManager import PitchExtractorManager
from voice_changer.RVC.pipeline.PipelineGenerator import createPipeline
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.pipeline.Pipeline import Pipeline

from Exceptions import DeviceCannotSupportHalfPrecisionException


class RVC(VoiceChangerModel):
    initialLoad: bool = True
    settings: RVCSettings = RVCSettings()

    pipeline: Pipeline | None = None

    deviceManager = DeviceManager.get_instance()

    audio_buffer: AudioInOut | None = None
    prevVol: float = 0
    params: VoiceChangerParams
    currentSlot: int = 0
    needSwitch: bool = False

    def __init__(self, params: VoiceChangerParams, slotInfo: RVCModelSlot):
        print("[Voice Changer] [RVC] Creating instance ")
        EmbedderManager.initialize(params)

        self.params = params
        self.pitchExtractor = PitchExtractorManager.getPitchExtractor(self.settings.f0Detector)

        self.prevVol = 0.0
        self.slotInfo = slotInfo
        self.initialize()

    def initialize(self):
        print("[Voice Changer] [RVC] Initializing... ")

        # pipelineの生成
        self.pipeline = createPipeline(self.slotInfo, self.settings.gpu, self.settings.f0Detector)

        # その他の設定
        self.settings.tran = self.slotInfo.defaultTune
        self.settings.indexRatio = self.slotInfo.defaultIndexRatio
        self.settings.protect = self.slotInfo.defaultProtect
        print("[Voice Changer] [RVC] Initializing... done")

    def update_settings(self, key: str, val: int | float | str):
        print("[Voice Changer][RVC]: update_settings", key, val)
        if key in self.settings.intData:
            setattr(self.settings, key, int(val))
            if key == "gpu":
                self.deviceManager.setForceTensor(False)
                self.initialize()
        elif key in self.settings.floatData:
            setattr(self.settings, key, float(val))
        elif key in self.settings.strData:
            setattr(self.settings, key, str(val))
            if key == "f0Detector" and self.pipeline is not None:
                pitchExtractor = PitchExtractorManager.getPitchExtractor(self.settings.f0Detector)
                self.pipeline.setPitchExtractor(pitchExtractor)
        else:
            return False
        return True

    def get_info(self):
        data = asdict(self.settings)
        if self.pipeline is not None:
            pipelineInfo = self.pipeline.getPipelineInfo()
            data["pipelineInfo"] = pipelineInfo
        return data

    def get_processing_sampling_rate(self):
        return self.slotInfo.samplingRate

    def generate_input(
        self,
        newData: AudioInOut,
        inputSize: int,
        crossfadeSize: int,
        solaSearchFrame: int = 0,
    ):
        newData = newData.astype(np.float32) / 32768.0  # RVCのモデルのサンプリングレートで入ってきている。（extraDataLength, Crossfade等も同じSRで処理）(★１)

        if self.audio_buffer is not None:
            # 過去のデータに連結
            self.audio_buffer = np.concatenate([self.audio_buffer, newData], 0)
        else:
            self.audio_buffer = newData

        convertSize = inputSize + crossfadeSize + solaSearchFrame + self.settings.extraConvertSize

        if convertSize % 128 != 0:  # モデルの出力のホップサイズで切り捨てが発生するので補う。
            convertSize = convertSize + (128 - (convertSize % 128))

        # バッファがたまっていない場合はzeroで補う
        if self.audio_buffer.shape[0] < convertSize:
            self.audio_buffer = np.concatenate([np.zeros([convertSize]), self.audio_buffer])

        convertOffset = -1 * convertSize
        self.audio_buffer = self.audio_buffer[convertOffset:]  # 変換対象の部分だけ抽出

        if self.pipeline is not None:
            device = self.pipeline.device
        else:
            device = torch.device("cpu")

        audio_buffer = torch.from_numpy(self.audio_buffer).to(device=device, dtype=torch.float32)

        # 出力部分だけ切り出して音量を確認。(TODO:段階的消音にする)
        cropOffset = -1 * (inputSize + crossfadeSize)
        cropEnd = -1 * (crossfadeSize)
        crop = audio_buffer[cropOffset:cropEnd]
        vol = torch.sqrt(torch.square(crop).mean()).detach().cpu().numpy()
        vol = max(vol, self.prevVol * 0.0)
        self.prevVol = vol

        return (audio_buffer, convertSize, vol)

    def inference(self, data):
        audio = data[0]
        convertSize = data[1]
        vol = data[2]

        if vol < self.settings.silentThreshold:
            return np.zeros(convertSize).astype(np.int16)

        audio = torchaudio.functional.resample(audio, self.slotInfo.samplingRate, 16000, rolloff=0.99)
        repeat = 1 if self.settings.rvcQuality else 0
        sid = 0
        f0_up_key = self.settings.tran
        index_rate = self.settings.indexRatio
        protect = self.settings.protect

        if_f0 = 1 if self.slotInfo.f0 else 0
        embOutputLayer = self.slotInfo.embOutputLayer
        useFinalProj = self.slotInfo.useFinalProj

        try:
            audio_out = self.pipeline.exec(
                sid,
                audio,
                f0_up_key,
                index_rate,
                if_f0,
                self.settings.extraConvertSize / self.slotInfo.samplingRate,  # extaraDataSizeの秒数。RVCのモデルのサンプリングレートで処理(★１)。
                embOutputLayer,
                useFinalProj,
                repeat,
                protect,
            )
            result = audio_out.detach().cpu().numpy() * np.sqrt(vol)

            return result
        except DeviceCannotSupportHalfPrecisionException as e:
            print("[Device Manager] Device cannot support half precision. Fallback to float....")
            self.deviceManager.setForceTensor(True)
            self.prepareModel(self.settings.modelSlotIndex)
            raise e

        return

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
                    # print("remove", key, file_path)
                    sys.modules.pop(key)
            except Exception:  # type:ignore
                # print(e)
                pass

    def export2onnx(self):
        allModelSlots = self.modelSlotManager.getAllSlotInfo()
        modelSlot = allModelSlots[self.settings.modelSlotIndex]

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
        # req: MergeModelRequest = MergeModelRequest.from_json(request)
        # merged = merge_model(req)
        # targetSlot = 0
        # if req.slot < 0:
        #     # 最後尾のスロット番号を格納先とする。
        #     allModelSlots = self.modelSlotManager.getAllSlotInfo()
        #     targetSlot = len(allModelSlots) - 1
        # else:
        #     targetSlot = req.slot

        # # いったんは、アップロードフォルダに格納する。（歴史的経緯）
        # # 後続のloadmodelを呼び出すことで永続化モデルフォルダに移動させられる。
        # storeDir = os.path.join(UPLOAD_DIR, f"{targetSlot}")
        # print("[Voice Changer] store merged model to:", storeDir)
        # os.makedirs(storeDir, exist_ok=True)
        # storeFile = os.path.join(storeDir, "merged.pth")
        # torch.save(merged, storeFile)

        # # loadmodelを呼び出して永続化モデルフォルダに移動させる。
        # params = {
        #     "defaultTune": req.defaultTune,
        #     "defaultIndexRatio": req.defaultIndexRatio,
        #     "defaultProtect": req.defaultProtect,
        #     "sampleId": "",
        #     "files": {"rvcModel": storeFile},
        # }
        # props: LoadModelParams = LoadModelParams(slot=targetSlot, isHalf=True, params=params)
        # self.loadModel(props)
        # self.prepareModel(targetSlot)
        # self.settings.modelSlotIndex = targetSlot
        # self.currentSlot = self.settings.modelSlotIndex

    def get_model_current(self):
        return [
            {
                "key": "defaultTune",
                "val": self.settings.tran,
            },
            {
                "key": "defaultIndexRatio",
                "val": self.settings.indexRatio,
            },
            {
                "key": "defaultProtect",
                "val": self.settings.protect,
            },
        ]
