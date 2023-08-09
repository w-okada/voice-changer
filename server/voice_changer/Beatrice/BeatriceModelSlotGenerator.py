import os

from data.ModelSlot import BeatriceModelSlot
from voice_changer.utils.LoadModelParams import LoadModelParams
from voice_changer.utils.ModelSlotGenerator import ModelSlotGenerator


class BeatriceModelSlotGenerator(ModelSlotGenerator):
    @classmethod
    def loadModel(cls, props: LoadModelParams):
        slotInfo: BeatriceModelSlot = BeatriceModelSlot()
        for file in props.files:
            if file.kind == "beatriceModel":
                slotInfo.modelFile = file.name
        slotInfo.name = os.path.splitext(os.path.basename(slotInfo.modelFile))[0]
        slotInfo.slotIndex = props.slot
        return slotInfo
