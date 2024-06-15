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
        inferencerType: str,
        file: str,
        inferencerTypeVersion: str | None = None,
    ) -> Inferencer:
        cls.currentInferencer = cls.loadInferencer(EnumInferenceTypes(inferencerType), file, inferencerTypeVersion)
        return cls.currentInferencer

    @classmethod
    def loadInferencer(
        cls,
        inferencerType: EnumInferenceTypes,
        file: str,
        inferencerTypeVersion: str | None = None,
    ) -> Inferencer:
        if inferencerType is EnumInferenceTypes.pyTorchRVC:
            return RVCInferencer().load_model(file)
        elif inferencerType is EnumInferenceTypes.pyTorchRVCNono:
            return RVCInferencerNono().load_model(file)
        elif inferencerType == EnumInferenceTypes.pyTorchRVCv2:
            return RVCInferencerv2().load_model(file)
        elif inferencerType is EnumInferenceTypes.pyTorchVoRASbeta:
            if sys.platform.startswith("darwin") is False:
                from voice_changer.RVC.inferencer.VorasInferencebeta import VoRASInferencer
                return VoRASInferencer().load_model(file)
            else:
                raise RuntimeError("[Voice Changer] VoRAS is not supported on macOS")
        elif inferencerType is EnumInferenceTypes.pyTorchRVCv2Nono:
            return RVCInferencerv2Nono().load_model(file)
        elif inferencerType is EnumInferenceTypes.pyTorchWebUI:
            return WebUIInferencer().load_model(file)
        elif inferencerType is EnumInferenceTypes.pyTorchWebUINono:
            return WebUIInferencerNono().load_model(file)
        elif inferencerType is EnumInferenceTypes.onnxRVC:
            return OnnxRVCInferencer().load_model(file, inferencerTypeVersion)
        elif inferencerType is EnumInferenceTypes.onnxRVCNono:
            return OnnxRVCInferencerNono().load_model(file, inferencerTypeVersion)
        else:
            raise RuntimeError("[Voice Changer] Inferencer not found", inferencerType)
