import os
from data.ModelSlot import DDSPSVCModelSlot
from voice_changer.utils.LoadModelParams import LoadModelParams
from voice_changer.utils.ModelSlotGenerator import ModelSlotGenerator


class DDSP_SVCModelSlotGenerator(ModelSlotGenerator):
    @classmethod
    def loadModel(cls, props: LoadModelParams):
        slotInfo: DDSPSVCModelSlot = DDSPSVCModelSlot()
        for file in props.files:
            if file.kind == "ddspSvcModelConfig":
                slotInfo.configFile = file.name
            elif file.kind == "ddspSvcModel":
                slotInfo.modelFile = file.name
            elif file.kind == "ddspSvcDiffusionConfig":
                slotInfo.diffConfigFile = file.name
            elif file.kind == "ddspSvcDiffusion":
                slotInfo.diffModelFile = file.name
        if slotInfo.configFile == '':
            slotInfo.configFile = 'builtin'
        if slotInfo.modelFile == '':
            slotInfo.modelFile = 'builtin'
        slotInfo.isONNX = slotInfo.modelFile.endswith(".onnx")
        slotInfo.name = os.path.splitext(os.path.basename(slotInfo.diffModelFile))[0]
        return slotInfo
