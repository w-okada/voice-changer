from const import EnumEmbedderTypes, EnumInferenceTypes
from voice_changer.RVC.ModelSlot import ModelSlot

from voice_changer.utils.LoadModelParams import FilePaths
import torch
import onnxruntime
import json


def generateModelSlot(files: FilePaths, params):
    modelSlot = ModelSlot()
    modelSlot.pyTorchModelFile = files.pyTorchModelFilename
    modelSlot.onnxModelFile = files.onnxModelFilename
    modelSlot.featureFile = files.featureFilename
    modelSlot.indexFile = files.indexFilename
    modelSlot.defaultTrans = params["trans"] if "trans" in params else 0

    modelSlot.isONNX = True if modelSlot.onnxModelFile is not None else False

    if modelSlot.isONNX:
        _setInfoByONNX(modelSlot, modelSlot.onnxModelFile)
    else:
        _setInfoByPytorch(modelSlot, modelSlot.pyTorchModelFile)
    return modelSlot


def _setInfoByPytorch(slot: ModelSlot, file: str):
    cpt = torch.load(file, map_location="cpu")
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

    slot.samplingRate = cpt["config"][-1]

    del cpt


def _setInfoByONNX(slot: ModelSlot, file: str):
    tmp_onnx_session = onnxruntime.InferenceSession(
        file, providers=["CPUExecutionProvider"]
    )
    modelmeta = tmp_onnx_session.get_modelmeta()
    try:
        metadata = json.loads(modelmeta.custom_metadata_map["metadata"])

        slot.modelType = metadata["modelType"]
        slot.embChannels = metadata["embChannels"]
        slot.embedder = (
            metadata["embedder"] if "embedder" in metadata else EnumEmbedderTypes.hubert
        )
        slot.f0 = metadata["f0"]
        slot.modelType = (
            EnumInferenceTypes.onnxRVC if slot.f0 else EnumInferenceTypes.onnxRVCNono
        )
        slot.samplingRate = metadata["samplingRate"]
        slot.deprecated = False

    except:
        slot.modelType = EnumInferenceTypes.onnxRVC
        slot.embChannels = 256
        slot.embedder = EnumEmbedderTypes.hubert
        slot.f0 = True
        slot.samplingRate = 48000
        slot.deprecated = True

        print("[Voice Changer] ############## !!!! CAUTION !!!! ####################")
        print("[Voice Changer] This onnxfie is depricated. Please regenerate onnxfile.")
        print("[Voice Changer] ############## !!!! CAUTION !!!! ####################")

    del tmp_onnx_session
