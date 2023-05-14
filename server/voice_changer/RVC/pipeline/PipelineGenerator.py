import os
import traceback
import numpy as np
import faiss

from voice_changer.RVC.ModelSlot import ModelSlot
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.embedder.EmbedderManager import EmbedderManager
from voice_changer.RVC.inferencer.InferencerManager import InferencerManager
from voice_changer.RVC.pipeline.Pipeline import Pipeline
from voice_changer.RVC.pitchExtractor.PitchExtractorManager import PitchExtractorManager


def createPipeline(modelSlot: ModelSlot, gpu: int, f0Detector: str):
    dev = DeviceManager.get_instance().getDevice(gpu)
    half = DeviceManager.get_instance().halfPrecisionAvailable(gpu)
    # # ファイル名特定(Inferencer)
    # inferencerFilename = (
    #     modelSlot.onnxModelFile if modelSlot.isONNX else modelSlot.pyTorchModelFile
    # )

    # Inferencer 生成
    try:
        inferencer = InferencerManager.getInferencer(
            modelSlot.modelType,
            modelSlot.modelFile,
            half,
            dev,
        )
    except Exception as e:
        print("[Voice Changer] exception! loading inferencer", e)
        traceback.print_exc()

    # Embedder 生成
    try:
        embedder = EmbedderManager.getEmbedder(
            modelSlot.embedder,
            # emmbedderFilename,
            half,
            dev,
        )
    except Exception as e:
        print("[Voice Changer]  exception! loading embedder", e)
        traceback.print_exc()

    # pitchExtractor
    pitchExtractor = PitchExtractorManager.getPitchExtractor(f0Detector)

    # index, feature
    index, feature = _loadIndex(modelSlot)

    pipeline = Pipeline(
        embedder,
        inferencer,
        pitchExtractor,
        index,
        feature,
        modelSlot.samplingRate,
        dev,
        half,
    )

    return pipeline


def _loadIndex(modelSlot: ModelSlot):
    # Indexのロード
    print("[Voice Changer] Loading index...")
    # ファイル指定がない場合はNone
    if modelSlot.featureFile is None or modelSlot.indexFile is None:
        print("[Voice Changer] Index is None, not used")
        return None, None

    # ファイル指定があってもファイルがない場合はNone
    if (
        os.path.exists(modelSlot.featureFile) is not True
        or os.path.exists(modelSlot.indexFile) is not True
    ):
        return None, None

    try:
        index = faiss.read_index(modelSlot.indexFile)
        feature = np.load(modelSlot.featureFile)
    except:
        print("[Voice Changer] load index failed. Use no index.")
        traceback.print_exc()
        return None, None

    return index, feature
