from typing import Protocol
from const import PitchExtractorType
from voice_changer.DiffusionSVC.pitchExtractor.CrepePitchExtractor import CrepePitchExtractor
from voice_changer.DiffusionSVC.pitchExtractor.DioPitchExtractor import DioPitchExtractor
from voice_changer.DiffusionSVC.pitchExtractor.HarvestPitchExtractor import HarvestPitchExtractor
from voice_changer.DiffusionSVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams


class PitchExtractorManager(Protocol):
    currentPitchExtractor: PitchExtractor | None = None
    params: VoiceChangerParams

    @classmethod
    def initialize(cls, params: VoiceChangerParams):
        cls.params = params

    @classmethod
    def getPitchExtractor(
        cls, pitchExtractorType: PitchExtractorType, gpu: int
    ) -> PitchExtractor:
        cls.currentPitchExtractor = cls.loadPitchExtractor(pitchExtractorType,  gpu)
        return cls.currentPitchExtractor

    @classmethod
    def loadPitchExtractor(
        cls, pitchExtractorType: PitchExtractorType, gpu: int
    ) -> PitchExtractor:
        if pitchExtractorType == "harvest":
            return HarvestPitchExtractor()
        elif pitchExtractorType == "dio":
            return DioPitchExtractor()
        elif pitchExtractorType == "crepe":
            return CrepePitchExtractor()
        # elif pitchExtractorType == "crepe_tiny":
        #     return CrepeOnnxPitchExtractor(pitchExtractorType, cls.params.crepe_onnx_tiny, gpu)
        # elif pitchExtractorType == "crepe_full":
        #     return CrepeOnnxPitchExtractor(pitchExtractorType, cls.params.crepe_onnx_full, gpu)
        else:
            # return hubert as default
            raise RuntimeError(
                "[Voice Changer] PitchExctractor not found", pitchExtractorType
            )
