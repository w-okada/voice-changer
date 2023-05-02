from typing import Protocol
from const import EnumPitchExtractorTypes


class PitchExtractor(Protocol):
    pitchExtractorType: EnumPitchExtractorTypes = EnumPitchExtractorTypes.harvest

    def extract(self, audio, f0_up_key, sr, window, silence_front=0):
        ...
