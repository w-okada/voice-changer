from const import DiffusionSVCInferenceType
from voice_changer.DiffusionSVC.inferencer.DiffusionSVCInferencer import DiffusionSVCInferencer
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
import os


class InferencerManager:
    currentInferencer: Inferencer | None = None
    params: VoiceChangerParams
    
    @classmethod
    def initialize(cls, params: VoiceChangerParams):
        cls.params = params

    @classmethod
    def getInferencer(
        cls,
        inferencerType: DiffusionSVCInferenceType,
        file: str,
        gpu: int,
    ) -> Inferencer:
        cls.currentInferencer = cls.loadInferencer(inferencerType, file, gpu)
        return cls.currentInferencer

    @classmethod
    def loadInferencer(
        cls,
        inferencerType: DiffusionSVCInferenceType,
        file: str,
        gpu: int,
    ) -> Inferencer:
        if inferencerType == "combo":
            vocoder_onnx_path = os.path.join(os.path.dirname(cls.params.nsf_hifigan), "nsf_hifigan.onnx")
            return DiffusionSVCInferencer(cls.params.nsf_hifigan, vocoder_onnx_path).loadModel(file, gpu)

        else:
            raise RuntimeError("[Voice Changer] Inferencer not found", inferencerType)
