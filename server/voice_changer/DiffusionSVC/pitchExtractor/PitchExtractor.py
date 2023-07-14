from typing import Protocol


class PitchExtractor(Protocol):

    def extract(self, audio, f0_up_key, sr, window, silence_front=0):
        ...

    def getPitchExtractorInfo(self):
        return {
            "pitchExtractorType": self.pitchExtractorType,
        }
