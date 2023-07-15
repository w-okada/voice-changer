import os
from const import EnumInferenceTypes
from dataclasses import asdict
import onnxruntime
import json

from data.ModelSlot import DiffusionSVCModelSlot, ModelSlot, RVCModelSlot
from voice_changer.DiffusionSVC.inferencer.diffusion_svc_model.diffusion.unit2mel import load_model_vocoder_from_combo
from voice_changer.utils.LoadModelParams import LoadModelParams
from voice_changer.utils.ModelSlotGenerator import ModelSlotGenerator


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
        slotInfo.iconFile = "/assets/icons/noimage.png"
        slotInfo.embChannels = 768

        if slotInfo.isONNX:
            slotInfo = cls._setInfoByONNX(slotInfo)
        else:
            slotInfo = cls._setInfoByPytorch(slotInfo)
        return slotInfo

    @classmethod
    def _setInfoByPytorch(cls, slot: DiffusionSVCModelSlot):
        diff_model, diff_args, naive_model, naive_args, vocoder = load_model_vocoder_from_combo(slot.modelFile, device="cpu")
        slot.kStepMax = diff_args.model.k_step_max
        return slot

    @classmethod
    def _setInfoByONNX(cls, slot: ModelSlot):
        tmp_onnx_session = onnxruntime.InferenceSession(slot.modelFile, providers=["CPUExecutionProvider"])
        modelmeta = tmp_onnx_session.get_modelmeta()
        try:
            slot = RVCModelSlot(**asdict(slot))
            metadata = json.loads(modelmeta.custom_metadata_map["metadata"])

            # slot.modelType = metadata["modelType"]
            slot.embChannels = metadata["embChannels"]

            slot.embOutputLayer = metadata["embOutputLayer"] if "embOutputLayer" in metadata else 9
            slot.useFinalProj = metadata["useFinalProj"] if "useFinalProj" in metadata else True if slot.embChannels == 256 else False

            if slot.embChannels == 256:
                slot.useFinalProj = True
            else:
                slot.useFinalProj = False

            # ONNXモデルの情報を表示
            if slot.embChannels == 256 and slot.embOutputLayer == 9 and slot.useFinalProj is True:
                print("[Voice Changer] ONNX Model: Official v1 like")
            elif slot.embChannels == 768 and slot.embOutputLayer == 12 and slot.useFinalProj is False:
                print("[Voice Changer] ONNX Model: Official v2 like")
            else:
                print(f"[Voice Changer] ONNX Model: ch:{slot.embChannels}, L:{slot.embOutputLayer}, FP:{slot.useFinalProj}")

            if "embedder" not in metadata:
                slot.embedder = "hubert_base"
            else:
                slot.embedder = metadata["embedder"]

            slot.f0 = metadata["f0"]
            slot.modelType = EnumInferenceTypes.onnxRVC.value if slot.f0 else EnumInferenceTypes.onnxRVCNono.value
            slot.samplingRate = metadata["samplingRate"]
            slot.deprecated = False

        except Exception as e:
            slot.modelType = EnumInferenceTypes.onnxRVC.value
            slot.embChannels = 256
            slot.embedder = "hubert_base"
            slot.f0 = True
            slot.samplingRate = 48000
            slot.deprecated = True

            print("[Voice Changer] setInfoByONNX", e)
            print("[Voice Changer] ############## !!!! CAUTION !!!! ####################")
            print("[Voice Changer] This onnxfie is depricated. Please regenerate onnxfile.")
            print("[Voice Changer] ############## !!!! CAUTION !!!! ####################")

        del tmp_onnx_session
        return slot
