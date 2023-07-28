from const import EnumInferenceTypes
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from voice_changer.RVC.inferencer.OnnxRVCInferencer import OnnxRVCInferencer
from voice_changer.RVC.inferencer.OnnxRVCInferencerNono import OnnxRVCInferencerNono
from voice_changer.RVC.inferencer.RVCInferencer import RVCInferencer
from voice_changer.RVC.inferencer.RVCInferencerNono import RVCInferencerNono
from voice_changer.RVC.inferencer.RVCInferencerv2 import RVCInferencerv2
from voice_changer.RVC.inferencer.RVCInferencerv2Nono import RVCInferencerv2Nono
from voice_changer.RVC.inferencer.WebUIInferencer import WebUIInferencer
from voice_changer.RVC.inferencer.WebUIInferencerNono import WebUIInferencerNono
import sys


class InferencerManager:
    currentInferencer: Inferencer | None = None

    @classmethod
    def getInferencer(
        cls,
        inferencerType: EnumInferenceTypes,
        file: str,
        gpu: int,
    ) -> Inferencer:
        cls.currentInferencer = cls.loadInferencer(inferencerType, file, gpu)
        return cls.currentInferencer

    @classmethod
    def loadInferencer(
        cls,
        inferencerType: EnumInferenceTypes,
        file: str,
        gpu: int,
    ) -> Inferencer:
        if inferencerType == EnumInferenceTypes.pyTorchRVC or inferencerType == EnumInferenceTypes.pyTorchRVC.value:
            return RVCInferencer().loadModel(file, gpu)
        elif inferencerType == EnumInferenceTypes.pyTorchRVCNono or inferencerType == EnumInferenceTypes.pyTorchRVCNono.value:
            return RVCInferencerNono().loadModel(file, gpu)
        elif inferencerType == EnumInferenceTypes.pyTorchRVCv2 or inferencerType == EnumInferenceTypes.pyTorchRVCv2.value:
            return RVCInferencerv2().loadModel(file, gpu)
        elif inferencerType == EnumInferenceTypes.pyTorchVoRASbeta or inferencerType == EnumInferenceTypes.pyTorchVoRASbeta.value:
            if sys.platform.startswith("darwin") is False:
                from voice_changer.RVC.inferencer.VorasInferencebeta import VoRASInferencer
                return VoRASInferencer().loadModel(file, gpu)
            else:
                raise RuntimeError("[Voice Changer] VoRAS is not supported on macOS")
        elif inferencerType == EnumInferenceTypes.pyTorchRVCv2Nono or inferencerType == EnumInferenceTypes.pyTorchRVCv2Nono.value:
            return RVCInferencerv2Nono().loadModel(file, gpu)
        elif inferencerType == EnumInferenceTypes.pyTorchWebUI or inferencerType == EnumInferenceTypes.pyTorchWebUI.value:
            return WebUIInferencer().loadModel(file, gpu)
        elif inferencerType == EnumInferenceTypes.pyTorchWebUINono or inferencerType == EnumInferenceTypes.pyTorchWebUINono.value:
            return WebUIInferencerNono().loadModel(file, gpu)
        elif inferencerType == EnumInferenceTypes.onnxRVC or inferencerType == EnumInferenceTypes.onnxRVC.value:
            return OnnxRVCInferencer().loadModel(file, gpu)
        elif inferencerType == EnumInferenceTypes.onnxRVCNono or inferencerType == EnumInferenceTypes.onnxRVCNono.value:
            return OnnxRVCInferencerNono().loadModel(file, gpu)
        else:
            raise RuntimeError("[Voice Changer] Inferencer not found", inferencerType)
