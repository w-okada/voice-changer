import os
from dataclasses import asdict


from data.ModelSlot import DiffusionSVCModelSlot, ModelSlot, RVCModelSlot
from voice_changer.DiffusionSVC.inferencer.diffusion_svc_model.diffusion.unit2mel import load_model_vocoder_from_combo
from voice_changer.VoiceChangerParamsManager import VoiceChangerParamsManager
from voice_changer.utils.LoadModelParams import LoadModelParams
from voice_changer.utils.ModelSlotGenerator import ModelSlotGenerator


def get_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5)+1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n //i)
    return sorted(divisors)


class DiffusionSVCModelSlotGenerator(ModelSlotGenerator):
    @classmethod
    def loadModel(cls, props: LoadModelParams):
        slotInfo: DiffusionSVCModelSlot = DiffusionSVCModelSlot()
        for file in props.files:
            if file.kind == "diffusionSVCModel":
                slotInfo.modelFile = file.name
        slotInfo.defaultTune = 0
        slotInfo.isONNX = slotInfo.modelFile.endswith(".onnx")
        slotInfo.name = os.path.splitext(os.path.basename(slotInfo.modelFile))[0]
        # slotInfo.iconFile = "/assets/icons/noimage.png"
        slotInfo.embChannels = 768
        slotInfo.slotIndex = props.slot

        if slotInfo.isONNX:
            slotInfo = cls._setInfoByONNX(slotInfo)
        else:
            slotInfo = cls._setInfoByPytorch(slotInfo)
        return slotInfo

    @classmethod
    def _setInfoByPytorch(cls, slot: DiffusionSVCModelSlot):
        vcparams = VoiceChangerParamsManager.get_instance().params
        modelPath = os.path.join(vcparams.model_dir, str(slot.slotIndex), os.path.basename(slot.modelFile))

        diff_model, diff_args, naive_model, naive_args = load_model_vocoder_from_combo(modelPath, device="cpu")
        slot.kStepMax = diff_args.model.k_step_max
        slot.nLayers = diff_args.model.n_layers
        slot.nnLayers = naive_args.model.n_layers
        slot.defaultKstep = slot.kStepMax
        divs = get_divisors(slot.defaultKstep)
        slot.defaultSpeedup = divs[-2]
        slot.speakers = {(x+1): f"user{x+1}" for x in range(diff_args.model.n_spk)}
        return slot

    @classmethod
    def _setInfoByONNX(cls, slot: ModelSlot):
        return slot
