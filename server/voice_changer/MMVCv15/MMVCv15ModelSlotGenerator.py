import os

from data.ModelSlot import MMVCv15ModelSlot
from voice_changer.utils.LoadModelParams import LoadModelParams
from voice_changer.utils.ModelSlotGenerator import ModelSlotGenerator


class MMVCv15ModelSlotGenerator(ModelSlotGenerator):
    @classmethod
    def loadModel(cls, props: LoadModelParams):
        slotInfo: MMVCv15ModelSlot = MMVCv15ModelSlot()
        for file in props.files:
            if file.kind == "mmvcv15Model":
                slotInfo.modelFile = file.name
            elif file.kind == "mmvcv15Config":
                slotInfo.configFile = file.name
        slotInfo.isONNX = slotInfo.modelFile.endswith(".onnx")
        slotInfo.name = os.path.splitext(os.path.basename(slotInfo.modelFile))[0]
        return slotInfo
