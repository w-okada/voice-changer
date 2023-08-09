from typing import Union
import os
import numpy as np
from data.ModelSlot import BeatriceModelSlot
from mods.log_control import VoiceChangaerLogger

from voice_changer.utils.VoiceChangerModel import AudioInOut, VoiceChangerModel
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams

from beatrice_internal_api import BeatriceInternalAPI

logger = VoiceChangaerLogger.get_instance().getLogger()


class BeatriceAPI(BeatriceInternalAPI):
    def __init__(self, sample_rate: float = 48000.0):
        pass


class Beatrice(VoiceChangerModel):
    def __init__(self, params: VoiceChangerParams, slotInfo: BeatriceModelSlot):
        raise RuntimeError("not implemented")

    def initialize(self):
        raise RuntimeError("not implemented")

    def setSamplingRate(self, inputSampleRate, outputSampleRate):
        raise RuntimeError("not implemented")

    def update_settings(self, key: str, val: int | float | str):
        raise RuntimeError("not implemented")

    def get_info(self):
        raise RuntimeError("not implemented")

    def get_processing_sampling_rate(self):
        raise RuntimeError("not implemented")

    def generate_input(
        self,
        newData: AudioInOut,
        crossfadeSize: int,
        solaSearchFrame: int = 0,
    ):
        raise RuntimeError("not implemented")

    def inference(self, receivedData: AudioInOut, crossfade_frame: int, sola_search_frame: int):
        raise RuntimeError("not implemented")

    def __del__(self):
        del self.pipeline

    def get_model_current(self):
        return [
            {
                "key": "dstId",
                "val": self.settings.dstId,
            },
        ]
