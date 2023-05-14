from const import EnumEmbedderTypes, EnumInferenceTypes
from voice_changer.RVC.ModelSlot import ModelSlot

import torch
import onnxruntime
import json
import os


def generateModelSlot_(params):
    modelSlot = ModelSlot()

    modelSlot.modelFile = params["files"]["rvcModel"]
    modelSlot.featureFile = (
        params["files"]["rvcFeature"] if "rvcFeature" in params["files"] else None
    )
    modelSlot.indexFile = (
        params["files"]["rvcIndex"] if "rvcIndex" in params["files"] else None
    )

    modelSlot.defaultTrans = params["trans"] if "trans" in params else 0

    modelSlot.isONNX = modelSlot.modelFile.endswith(".onnx")

    if modelSlot.isONNX:
        _setInfoByONNX(modelSlot)
    else:
        _setInfoByPytorch(modelSlot)
    return modelSlot


def generateModelSlot(slotDir: str):
    modelSlot = ModelSlot()
    if os.path.exists(slotDir) == False:
        return modelSlot
    paramFile = os.path.join(slotDir, "params.json")
    with open(paramFile, "r") as f:
        params = json.load(f)

    modelSlot.modelFile = os.path.join(
        slotDir, os.path.basename(params["files"]["rvcModel"])
    )
    if "rvcFeature" in params["files"]:
        modelSlot.featureFile = os.path.join(
            slotDir, os.path.basename(params["files"]["rvcFeature"])
        )
    else:
        modelSlot.featureFile = None
    if "rvcIndex" in params["files"]:
        modelSlot.indexFile = os.path.join(
            slotDir, os.path.basename(params["files"]["rvcIndex"])
        )
    else:
        modelSlot.indexFile = None

    modelSlot.defaultTrans = params["trans"] if "trans" in params else 0

    modelSlot.isONNX = modelSlot.modelFile.endswith(".onnx")

    if modelSlot.isONNX:
        _setInfoByONNX(modelSlot)
    else:
        _setInfoByPytorch(modelSlot)
    return modelSlot


def _setInfoByPytorch(slot: ModelSlot):
    cpt = torch.load(slot.modelFile, map_location="cpu")
    config_len = len(cpt["config"])
    if config_len == 18:
        slot.f0 = True if cpt["f0"] == 1 else False
        slot.modelType = (
            EnumInferenceTypes.pyTorchRVC
            if slot.f0
            else EnumInferenceTypes.pyTorchRVCNono
        )
        slot.embChannels = 256
        slot.embedder = EnumEmbedderTypes.hubert
    else:
        slot.f0 = True if cpt["f0"] == 1 else False
        slot.modelType = (
            EnumInferenceTypes.pyTorchWebUI
            if slot.f0
            else EnumInferenceTypes.pyTorchWebUINono
        )
        slot.embChannels = cpt["config"][17]
        slot.embedder = cpt["embedder_name"]
        if slot.embedder.endswith("768"):
            slot.embedder = slot.embedder[:-3]

        if slot.embedder == EnumEmbedderTypes.hubert.value:
            slot.embedder = EnumEmbedderTypes.hubert
        elif slot.embedder == EnumEmbedderTypes.contentvec.value:
            slot.embedder = EnumEmbedderTypes.contentvec
        elif slot.embedder == EnumEmbedderTypes.hubert_jp.value:
            slot.embedder = EnumEmbedderTypes.hubert_jp
        else:
            raise RuntimeError("[Voice Changer][setInfoByONNX] unknown embedder")

    slot.samplingRate = cpt["config"][-1]

    del cpt


def _setInfoByONNX(slot: ModelSlot):
    tmp_onnx_session = onnxruntime.InferenceSession(
        slot.modelFile, providers=["CPUExecutionProvider"]
    )
    modelmeta = tmp_onnx_session.get_modelmeta()
    try:
        metadata = json.loads(modelmeta.custom_metadata_map["metadata"])

        # slot.modelType = metadata["modelType"]
        slot.embChannels = metadata["embChannels"]

        if "embedder" not in metadata:
            slot.embedder = EnumEmbedderTypes.hubert
        elif metadata["embedder"] == EnumEmbedderTypes.hubert.value:
            slot.embedder = EnumEmbedderTypes.hubert
        elif metadata["embedder"] == EnumEmbedderTypes.contentvec.value:
            slot.embedder = EnumEmbedderTypes.contentvec
        elif metadata["embedder"] == EnumEmbedderTypes.hubert_jp.value:
            slot.embedder = EnumEmbedderTypes.hubert_jp
        else:
            raise RuntimeError("[Voice Changer][setInfoByONNX] unknown embedder")

        slot.f0 = metadata["f0"]
        slot.modelType = (
            EnumInferenceTypes.onnxRVC if slot.f0 else EnumInferenceTypes.onnxRVCNono
        )
        slot.samplingRate = metadata["samplingRate"]
        slot.deprecated = False

    except Exception as e:
        slot.modelType = EnumInferenceTypes.onnxRVC
        slot.embChannels = 256
        slot.embedder = EnumEmbedderTypes.hubert
        slot.f0 = True
        slot.samplingRate = 48000
        slot.deprecated = True

        print("[Voice Changer] setInfoByONNX", e)
        print("[Voice Changer] ############## !!!! CAUTION !!!! ####################")
        print("[Voice Changer] This onnxfie is depricated. Please regenerate onnxfile.")
        print("[Voice Changer] ############## !!!! CAUTION !!!! ####################")

    del tmp_onnx_session
