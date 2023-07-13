from const import DiffusionSVCInferenceType
from voice_changer.DiffusionSVC.inferencer.DiffusionSVCInferencer import DiffusionSVCInferencer
from voice_changer.RVC.inferencer.Inferencer import Inferencer


class InferencerManager:
    currentInferencer: Inferencer | None = None

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
            return DiffusionSVCInferencer().loadModel(file, gpu)
        else:
            raise RuntimeError("[Voice Changer] Inferencer not found", inferencerType)
