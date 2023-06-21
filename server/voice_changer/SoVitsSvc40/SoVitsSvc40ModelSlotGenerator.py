import os

from data.ModelSlot import SoVitsSvc40ModelSlot
from voice_changer.utils.LoadModelParams import LoadModelParams
from voice_changer.utils.ModelSlotGenerator import ModelSlotGenerator


class SoVitsSvc40ModelSlotGenerator(ModelSlotGenerator):
    @classmethod
    def loadModel(cls, props: LoadModelParams):
        slotInfo: SoVitsSvc40ModelSlot = SoVitsSvc40ModelSlot()
        for file in props.files:
            if file.kind == "soVitsSvc40Config":
                slotInfo.configFile = file.name
            elif file.kind == "soVitsSvc40Model":
                slotInfo.modelFile = file.name
            elif file.kind == "soVitsSvc40Cluster":
                slotInfo.clusterFile = file.name
        slotInfo.isONNX = slotInfo.modelFile.endswith(".onnx")
        slotInfo.name = os.path.splitext(os.path.basename(slotInfo.modelFile))[0]
        return slotInfo
