from typing import Protocol
from const import EnumPitchExtractorTypes
from voice_changer.RVC.pitchExtractor.DioPitchExtractor import DioPitchExtractor
from voice_changer.RVC.pitchExtractor.HarvestPitchExtractor import HarvestPitchExtractor
from voice_changer.RVC.pitchExtractor.CrepePitchExtractor import CrepePitchExtractor
from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor


class PitchExtractorManager(Protocol):
    currentPitchExtractor: PitchExtractor | None = None

    @classmethod
    def getPitchExtractor(
        cls, pitchExtractorType: EnumPitchExtractorTypes
    ) -> PitchExtractor:
        cls.currentPitchExtractor = cls.loadPitchExtractor(pitchExtractorType)
        return cls.currentPitchExtractor

    @classmethod
    def loadPitchExtractor(
        cls, pitchExtractorType: EnumPitchExtractorTypes
    ) -> PitchExtractor:
        if (
            pitchExtractorType == EnumPitchExtractorTypes.harvest
            or pitchExtractorType == EnumPitchExtractorTypes.harvest.value
        ):
            return HarvestPitchExtractor()
        elif (
            pitchExtractorType == EnumPitchExtractorTypes.dio
            or pitchExtractorType == EnumPitchExtractorTypes.dio.value
        ):
            return DioPitchExtractor()
        elif (
            pitchExtractorType == EnumPitchExtractorTypes.crepe
            or pitchExtractorType == EnumPitchExtractorTypes.crepe.value
        ):
            return CrepePitchExtractor()
        else:
            # return hubert as default
            raise RuntimeError(
                "[Voice Changer] PitchExctractor not found", pitchExtractorType
            )
