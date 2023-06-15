from const import UPLOAD_DIR, EnumEmbedderTypes, EnumInferenceTypes

import torch
import onnxruntime
import json
import os
import shutil
from data.ModelSlot import ModelSlot, RVCModelSlot, saveSlotInfo


def setSlotAsRVC(model_dir: str, slot: int, paramDict):
    slotInfo: RVCModelSlot = RVCModelSlot()
    slotDir = os.path.join(model_dir, str(slot))
    os.makedirs(slotDir, exist_ok=True)

    print("RVC SLot Load", slot, paramDict)
    for f in paramDict["files"]:
        srcPath = os.path.join(UPLOAD_DIR, f["name"])
        dstPath = os.path.join(slotDir, f["name"])
        if f["kind"] == "rvcModel":
            slotInfo.modelFile = dstPath
            slotInfo.name = os.path.splitext(f["name"])[0]
        elif f["kind"] == "rvcIndex":
            slotInfo.indexFile = dstPath
        else:
            print(f"[Voice Changer] unknown file kind {f['kind']}")

        shutil.move(srcPath, dstPath)

    slotInfo.defaultTune = 0
    slotInfo.defaultIndexRatio = 1
    slotInfo.defaultProtect = 0.5
    slotInfo.isONNX = slotInfo.modelFile.endswith(".onnx")

    if slotInfo.isONNX:
        _setInfoByONNX(slotInfo)
    else:
        _setInfoByPytorch(slotInfo)

    saveSlotInfo(model_dir, slot, slotInfo)

    print("[Voice Changer] new model added:", slotInfo)


def _setInfoByPytorch(slot: ModelSlot):
    cpt = torch.load(slot.modelFile, map_location="cpu")
    config_len = len(cpt["config"])

    if config_len == 18:
        # Original RVC
        slot.f0 = True if cpt["f0"] == 1 else False
        version = cpt.get("version", "v1")
        if version is None or version == "v1":
            slot.modelType = EnumInferenceTypes.pyTorchRVC.value if slot.f0 else EnumInferenceTypes.pyTorchRVCNono.value
            slot.embChannels = 256
            slot.embOutputLayer = 9
            slot.useFinalProj = True
            slot.embedder = EnumEmbedderTypes.hubert.value
            print("[Voice Changer] Official Model(pyTorch) : v1")
        else:
            slot.modelType = EnumInferenceTypes.pyTorchRVCv2.value if slot.f0 else EnumInferenceTypes.pyTorchRVCv2Nono.value
            slot.embChannels = 768
            slot.embOutputLayer = 12
            slot.useFinalProj = False
            slot.embedder = EnumEmbedderTypes.hubert.value
            print("[Voice Changer] Official Model(pyTorch) : v2")

    else:
        # DDPN RVC
        slot.f0 = True if cpt["f0"] == 1 else False
        slot.modelType = EnumInferenceTypes.pyTorchWebUI.value if slot.f0 else EnumInferenceTypes.pyTorchWebUINono.value
        slot.embChannels = cpt["config"][17]
        slot.embOutputLayer = cpt["embedder_output_layer"] if "embedder_output_layer" in cpt else 9
        if slot.embChannels == 256:
            slot.useFinalProj = True
        else:
            slot.useFinalProj = False

        # DDPNモデルの情報を表示
        if slot.embChannels == 256 and slot.embOutputLayer == 9 and slot.useFinalProj is True:
            print("[Voice Changer] DDPN Model(pyTorch) : Official v1 like")
        elif slot.embChannels == 768 and slot.embOutputLayer == 12 and slot.useFinalProj is False:
            print("[Voice Changer] DDPN Model(pyTorch): Official v2 like")
        else:
            print(f"[Voice Changer] DDPN Model(pyTorch): ch:{slot.embChannels}, L:{slot.embOutputLayer}, FP:{slot.useFinalProj}")

        slot.embedder = cpt["embedder_name"]
        if slot.embedder.endswith("768"):
            slot.embedder = slot.embedder[:-3]

        # if slot.embedder == EnumEmbedderTypes.hubert.value:
        #     slot.embedder = EnumEmbedderTypes.hubert
        # elif slot.embedder == EnumEmbedderTypes.contentvec.value:
        #     slot.embedder = EnumEmbedderTypes.contentvec
        # elif slot.embedder == EnumEmbedderTypes.hubert_jp.value:
        #     slot.embedder = EnumEmbedderTypes.hubert_jp
        # else:
        #     raise RuntimeError("[Voice Changer][setInfoByONNX] unknown embedder")

    slot.samplingRate = cpt["config"][-1]

    del cpt


def _setInfoByONNX(slot: ModelSlot):
    print("......................................_setInfoByONNX")
    tmp_onnx_session = onnxruntime.InferenceSession(slot.modelFile, providers=["CPUExecutionProvider"])
    modelmeta = tmp_onnx_session.get_modelmeta()
    try:
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
            slot.embedder = EnumEmbedderTypes.hubert.value
        else:
            slot.embedder = metadata["embedder"]
        # elif metadata["embedder"] == EnumEmbedderTypes.hubert.value:
        #     slot.embedder = EnumEmbedderTypes.hubert
        # elif metadata["embedder"] == EnumEmbedderTypes.contentvec.value:
        #     slot.embedder = EnumEmbedderTypes.contentvec
        # elif metadata["embedder"] == EnumEmbedderTypes.hubert_jp.value:
        #     slot.embedder = EnumEmbedderTypes.hubert_jp
        # else:
        #     raise RuntimeError("[Voice Changer][setInfoByONNX] unknown embedder")

        slot.f0 = metadata["f0"]
        print("slot.modelType1", slot.modelType)
        slot.modelType = EnumInferenceTypes.onnxRVC.value if slot.f0 else EnumInferenceTypes.onnxRVCNono.value
        print("slot.modelType2", slot.modelType)
        slot.samplingRate = metadata["samplingRate"]
        slot.deprecated = False

    except Exception as e:
        slot.modelType = EnumInferenceTypes.onnxRVC.value
        slot.embChannels = 256
        slot.embedder = EnumEmbedderTypes.hubert.value
        slot.f0 = True
        slot.samplingRate = 48000
        slot.deprecated = True

        print("[Voice Changer] setInfoByONNX", e)
        print("[Voice Changer] ############## !!!! CAUTION !!!! ####################")
        print("[Voice Changer] This onnxfie is depricated. Please regenerate onnxfile.")
        print("[Voice Changer] ############## !!!! CAUTION !!!! ####################")

    del tmp_onnx_session
