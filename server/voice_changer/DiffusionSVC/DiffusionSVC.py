from dataclasses import asdict
import numpy as np
from data.ModelSlot import DiffusionSVCModelSlot
from voice_changer.DiffusionSVC.DiffusionSVCSettings import DiffusionSVCSettings
from voice_changer.DiffusionSVC.inferencer.InferencerManager import InferencerManager
from voice_changer.DiffusionSVC.pipeline.Pipeline import Pipeline
from voice_changer.DiffusionSVC.pipeline.PipelineGenerator import createPipeline
from voice_changer.DiffusionSVC.pitchExtractor.PitchExtractorManager import PitchExtractorManager

from voice_changer.utils.VoiceChangerModel import AudioInOut, PitchfInOut, FeatureInOut, VoiceChangerModel
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
from voice_changer.RVC.embedder.EmbedderManager import EmbedderManager
# from voice_changer.RVC.onnxExporter.export2onnx import export2onnx
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager

from Exceptions import DeviceCannotSupportHalfPrecisionException


class DiffusionSVC(VoiceChangerModel):
    def __init__(self, params: VoiceChangerParams, slotInfo: DiffusionSVCModelSlot):
        print("[Voice Changer] [DiffusionSVC] Creating instance ")
        self.deviceManager = DeviceManager.get_instance()
        EmbedderManager.initialize(params)
        PitchExtractorManager.initialize(params)
        InferencerManager.initialize(params)
        self.settings = DiffusionSVCSettings()
        self.params = params
        self.pitchExtractor = PitchExtractorManager.getPitchExtractor(self.settings.f0Detector, self.settings.gpu)

        self.pipeline: Pipeline | None = None

        self.audio_buffer: AudioInOut | None = None
        self.pitchf_buffer: PitchfInOut | None = None
        self.feature_buffer: FeatureInOut | None = None
        self.prevVol = 0.0
        self.slotInfo = slotInfo

    def initialize(self):
        print("[Voice Changer] [DiffusionSVC] Initializing... ")

        # pipelineの生成
        self.pipeline = createPipeline(self.slotInfo, self.settings.gpu, self.settings.f0Detector, self.inputSampleRate, self.outputSampleRate)

        # その他の設定
        self.settings.tran = self.slotInfo.defaultTune
        self.settings.dstId = self.slotInfo.dstId
        self.settings.kStep = self.slotInfo.defaultKstep

        print("[Voice Changer] [DiffusionSVC] Initializing... done")

    def setSamplingRate(self, inputSampleRate, outputSampleRate):
        self.inputSampleRate = inputSampleRate
        self.outputSampleRate = outputSampleRate
        self.initialize()

    def update_settings(self, key: str, val: int | float | str):
        print("[Voice Changer][DiffusionSVC]: update_settings", key, val)
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
                pitchExtractor = PitchExtractorManager.getPitchExtractor(self.settings.f0Detector, self.settings.gpu)
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
        crossfadeSize: int,
        solaSearchFrame: int = 0,
    ):
        newData = newData.astype(np.float32) / 32768.0  # DiffusionSVCのモデルのサンプリングレートで入ってきている。（extraDataLength, Crossfade等も同じSRで処理）(★１)
        new_feature_length = int(((newData.shape[0] / self.inputSampleRate) * self.slotInfo.samplingRate) / 512)  # 100 は hubertのhosizeから (16000 / 160).
        # ↑newData.shape[0]//sampleRate でデータ秒数。これに16000かけてhubertの世界でのデータ長。これにhop数(160)でわるとfeatsのデータサイズになる。
        if self.audio_buffer is not None:
            # 過去のデータに連結
            self.audio_buffer = np.concatenate([self.audio_buffer, newData], 0)
            self.pitchf_buffer = np.concatenate([self.pitchf_buffer, np.zeros(new_feature_length)], 0)
            self.feature_buffer = np.concatenate([self.feature_buffer, np.zeros([new_feature_length, self.slotInfo.embChannels])], 0)
        else:
            self.audio_buffer = newData
            self.pitchf_buffer = np.zeros(new_feature_length)
            self.feature_buffer = np.zeros([new_feature_length, self.slotInfo.embChannels])

        convertSize = newData.shape[0] + crossfadeSize + solaSearchFrame + self.settings.extraConvertSize

        if convertSize % 128 != 0:  # モデルの出力のホップサイズで切り捨てが発生するので補う。
            convertSize = convertSize + (128 - (convertSize % 128))

        # バッファがたまっていない場合はzeroで補う
        generateFeatureLength = int(((convertSize / self.inputSampleRate) * self.slotInfo.samplingRate) / 512) + 1
        if self.audio_buffer.shape[0] < convertSize:
            self.audio_buffer = np.concatenate([np.zeros([convertSize]), self.audio_buffer])
            self.pitchf_buffer = np.concatenate([np.zeros(generateFeatureLength), self.pitchf_buffer])
            self.feature_buffer = np.concatenate([np.zeros([generateFeatureLength, self.slotInfo.embChannels]), self.feature_buffer])

        convertOffset = -1 * convertSize
        featureOffset = -1 * generateFeatureLength
        self.audio_buffer = self.audio_buffer[convertOffset:]  # 変換対象の部分だけ抽出
        self.pitchf_buffer = self.pitchf_buffer[featureOffset:]
        self.feature_buffer = self.feature_buffer[featureOffset:]

        # 出力部分だけ切り出して音量を確認。(TODO:段階的消音にする)
        cropOffset = -1 * (newData.shape[0] + crossfadeSize)
        cropEnd = -1 * (crossfadeSize)
        crop = self.audio_buffer[cropOffset:cropEnd]
        vol = np.sqrt(np.square(crop).mean())
        vol = float(max(vol, self.prevVol * 0.0))
        self.prevVol = vol

        return (self.audio_buffer, self.pitchf_buffer, self.feature_buffer, convertSize, vol)

    def inference(self, receivedData: AudioInOut, crossfade_frame: int, sola_search_frame: int):
        data = self.generate_input(receivedData, crossfade_frame, sola_search_frame)
        audio: AudioInOut = data[0]
        pitchf: PitchfInOut = data[1]
        feature: FeatureInOut = data[2]
        convertSize: int = data[3]
        vol: float = data[4]

        if vol < self.settings.silentThreshold:
            return np.zeros(convertSize).astype(np.int16) * np.sqrt(vol)

        if self.pipeline is None:
            return np.zeros(convertSize).astype(np.int16) * np.sqrt(vol)

        # device = self.pipeline.device
        # audio = torch.from_numpy(audio).to(device=device, dtype=torch.float32)
        # audio = self.resampler16K(audio)
        sid = self.settings.dstId
        f0_up_key = self.settings.tran
        protect = 0

        kStep = self.settings.kStep
        speedUp = self.settings.speedUp
        embOutputLayer = 12
        useFinalProj = False
        silenceFrontSec = self.settings.extraConvertSize / self.inputSampleRate if self.settings.silenceFront else 0.  # extaraConvertSize(既にモデルのサンプリングレートにリサンプリング済み)の秒数。モデルのサンプリングレートで処理(★１)。

        try:
            audio_out, self.pitchf_buffer, self.feature_buffer = self.pipeline.exec(
                sid,
                audio,
                self.inputSampleRate,
                pitchf,
                feature,
                f0_up_key,
                kStep,
                speedUp,
                silenceFrontSec,
                embOutputLayer,
                useFinalProj,
                protect
            )
            result = audio_out.detach().cpu().numpy()
            return result
        except DeviceCannotSupportHalfPrecisionException as e:  # NOQA
            print("[Device Manager] Device cannot support half precision. Fallback to float....")
            self.deviceManager.setForceTensor(True)
            self.initialize()
            # raise e

        return

    def __del__(self):
        del self.pipeline

    # def export2onnx(self):
    #     modelSlot = self.slotInfo

    #     if modelSlot.isONNX:
    #         print("[Voice Changer] export2onnx, No pyTorch filepath.")
    #         return {"status": "ng", "path": ""}

    #     output_file_simple = export2onnx(self.settings.gpu, modelSlot)
    #     return {
    #         "status": "ok",
    #         "path": f"/tmp/{output_file_simple}",
    #         "filename": output_file_simple,
    #     }

    def get_model_current(self):
        return [
            {
                "key": "defaultTune",
                "val": self.settings.tran,
            },
            {
                "key": "defaultKstep",
                "val": self.settings.kStep,
            },
            {
                "key": "defaultSpeedup",
                "val": self.settings.speedUp,
            },
        ]
