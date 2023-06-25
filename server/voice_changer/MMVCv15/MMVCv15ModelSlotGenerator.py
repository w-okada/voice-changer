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
            elif file.kind == "mmvcv15Correspondence":
                with open(file.name, "r") as f:
                    slotInfo.speakers = {}
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        vals = line.strip().split("|")
                        if len(vals) != 3:
                            break
                        id = vals[0]
                        f0 = vals[1]
                        name = vals[2]
                        slotInfo.speakers[id] = name
                        slotInfo.f0[id] = f0

        slotInfo.isONNX = slotInfo.modelFile.endswith(".onnx")
        slotInfo.name = os.path.splitext(os.path.basename(slotInfo.modelFile))[0]
        return slotInfo
