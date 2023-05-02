from torch import device

from const import EnumInferenceTypes
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
    ) -> Inferencer:
        if (
            inferencerType == EnumInferenceTypes.pyTorchRVC
            or inferencerType == EnumInferenceTypes.pyTorchRVC.value
        ):
            return RVCInferencer().loadModel(file, dev, isHalf)
        elif (
            inferencerType == EnumInferenceTypes.pyTorchRVCNono
            or inferencerType == EnumInferenceTypes.pyTorchRVCNono.value
        ):
            return RVCInferencerNono().loadModel(file, dev, isHalf)
        elif (
            inferencerType == EnumInferenceTypes.pyTorchWebUI
            or inferencerType == EnumInferenceTypes.pyTorchWebUI.value
        ):
            return WebUIInferencer().loadModel(file, dev, isHalf)
        elif (
            inferencerType == EnumInferenceTypes.pyTorchWebUINono
            or inferencerType == EnumInferenceTypes.pyTorchWebUINono.value
        ):
            return WebUIInferencerNono().loadModel(file, dev, isHalf)
        elif (
            inferencerType == EnumInferenceTypes.onnxRVC
            or inferencerType == EnumInferenceTypes.onnxRVC.value
        ):
            return OnnxRVCInference().loadModel(file, dev, isHalf)
        elif (
            inferencerType == EnumInferenceTypes.onnxRVCNono
            or inferencerType == EnumInferenceTypes.onnxRVCNono.value
        ):
            return OnnxRVCInferenceNono().loadModel(file, dev, isHalf)
        else:
            raise RuntimeError("[Voice Changer] Inferencer not found", inferencerType)
