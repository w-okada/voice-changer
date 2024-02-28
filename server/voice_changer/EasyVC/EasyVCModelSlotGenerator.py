import os

from data.ModelSlot import EasyVCModelSlot
from voice_changer.utils.LoadModelParams import LoadModelParams
from voice_changer.utils.ModelSlotGenerator import ModelSlotGenerator


class EasyVCModelSlotGenerator(ModelSlotGenerator):
    @classmethod
    def loadModel(cls, props: LoadModelParams):
        slotInfo: EasyVCModelSlot = EasyVCModelSlot()
        for file in props.files:
            if file.kind == "easyVCModel":
                slotInfo.modelFile = file.name
        slotInfo.name = os.path.splitext(os.path.basename(slotInfo.modelFile))[0]
        slotInfo.slotIndex = props.slot
        return slotInfo
