import os

from data.ModelSlot import BeatriceModelSlot, LLVCModelSlot
from voice_changer.utils.LoadModelParams import LoadModelParams
from voice_changer.utils.ModelSlotGenerator import ModelSlotGenerator


class LLVCModelSlotGenerator(ModelSlotGenerator):
    @classmethod
    def loadModel(cls, props: LoadModelParams):
        slotInfo: LLVCModelSlot = LLVCModelSlot()
        for file in props.files:
            if file.kind == "llvcModel":
                slotInfo.modelFile = file.name
            if file.kind == "llvcConfig":
                slotInfo.configFile = file.name
        slotInfo.name = os.path.splitext(os.path.basename(slotInfo.modelFile))[0]
        slotInfo.slotIndex = props.slot
        return slotInfo
