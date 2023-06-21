import os

from data.ModelSlot import MMVCv13ModelSlot
from voice_changer.utils.LoadModelParams import LoadModelParams
from voice_changer.utils.ModelSlotGenerator import ModelSlotGenerator


class MMVCv13ModelSlotGenerator(ModelSlotGenerator):
    @classmethod
    def loadModel(cls, props: LoadModelParams):
        slotInfo: MMVCv13ModelSlot = MMVCv13ModelSlot()
        for file in props.files:
            if file.kind == "mmvcv13Model":
                slotInfo.modelFile = file.name
            elif file.kind == "mmvcv13Config":
                slotInfo.configFile = file.name
        slotInfo.isONNX = slotInfo.modelFile.endswith(".onnx")
        slotInfo.name = os.path.splitext(os.path.basename(slotInfo.modelFile))[0]
        return slotInfo
