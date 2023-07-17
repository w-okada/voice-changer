from typing import Protocol

from voice_changer.utils.VoiceChangerModel import AudioInOut


class PitchExtractor(Protocol):

    def extract(self, audio: AudioInOut, sr: int, block_size: int, model_sr: int, pitch, f0_up_key, silence_front=0):
        ...

    def getPitchExtractorInfo(self):
        return {
            "pitchExtractorType": self.pitchExtractorType,
        }
