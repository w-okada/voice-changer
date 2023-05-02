from torch import device

from const import EnumInferenceTypes
from voice_changer.RVC.embedder.Embedder import Embedder
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from voice_changer.RVC.inferencer.OnnxRVCInferencer import OnnxRVCInference
from voice_changer.RVC.inferencer.OnnxRVCInferencerNono import OnnxRVCInferenceNono
from voice_changer.RVC.inferencer.RVCInferencer import RVCInferencer
from voice_changer.RVC.inferencer.RVCInferencerNono import RVCInferencerNono
from voice_changer.RVC.inferencer.WebUIInferencer import WebUIInferencer
from voice_changer.RVC.inferencer.WebUIInferencerNono import WebUIInferencerNono


class InferencerManager:
    currentInferencer: Inferencer | None = None

    @classmethod
    def getInferencer(
        cls, inferencerType: EnumInferenceTypes, file: str, isHalf: bool, dev: device
    ) -> Inferencer:
        cls.currentInferencer = cls.loadInferencer(inferencerType, file, isHalf, dev)
        return cls.currentInferencer

    @classmethod
    def loadInferencer(
        cls, inferencerType: EnumInferenceTypes, file: str, isHalf: bool, dev: device
    ) -> Embedder:
        if inferencerType == EnumInferenceTypes.pyTorchRVC:
            return RVCInferencer().loadModel(file, dev, isHalf)
        elif inferencerType == EnumInferenceTypes.pyTorchRVCNono:
            return RVCInferencerNono().loadModel(file, dev, isHalf)
        elif inferencerType == EnumInferenceTypes.pyTorchWebUI:
            return WebUIInferencer().loadModel(file, dev, isHalf)
        elif inferencerType == EnumInferenceTypes.pyTorchWebUINono:
            return WebUIInferencerNono().loadModel(file, dev, isHalf)
        elif inferencerType == EnumInferenceTypes.onnxRVC:
            return OnnxRVCInference().loadModel(file, dev, isHalf)
        elif inferencerType == EnumInferenceTypes.onnxRVCNono:
            return OnnxRVCInferenceNono().loadModel(file, dev, isHalf)
        else:
            # return hubert as default
            raise RuntimeError("[Voice Changer] Inferencer not found", inferencerType)
