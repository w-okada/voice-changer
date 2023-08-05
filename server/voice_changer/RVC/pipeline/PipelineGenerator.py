import os
import traceback
import faiss
from Exceptions import PipelineCreateException
from data.ModelSlot import RVCModelSlot

from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.embedder.EmbedderManager import EmbedderManager
from voice_changer.RVC.inferencer.InferencerManager import InferencerManager
from voice_changer.RVC.pipeline.Pipeline import Pipeline
from voice_changer.RVC.pitchExtractor.PitchExtractorManager import PitchExtractorManager
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams


def createPipeline(params: VoiceChangerParams, modelSlot: RVCModelSlot, gpu: int, f0Detector: str):
    dev = DeviceManager.get_instance().getDevice(gpu)
    half = DeviceManager.get_instance().halfPrecisionAvailable(gpu)

    # Inferencer 生成
    try:
        modelPath = os.path.join(params.model_dir, str(modelSlot.slotIndex), os.path.basename(modelSlot.modelFile))
        inferencer = InferencerManager.getInferencer(modelSlot.modelType, modelPath, gpu)
    except Exception as e:
        print("[Voice Changer] exception! loading inferencer", e)
        traceback.print_exc()
        raise PipelineCreateException("[Voice Changer] exception! loading inferencer")

    # Embedder 生成
    try:
        embedder = EmbedderManager.getEmbedder(
            modelSlot.embedder,
            # emmbedderFilename,
            half,
            dev,
        )
    except Exception as e:
        print("[Voice Changer] exception! loading embedder", e, dev)
        traceback.print_exc()
        raise PipelineCreateException("[Voice Changer] exception! loading embedder")

    # pitchExtractor
    pitchExtractor = PitchExtractorManager.getPitchExtractor(f0Detector, gpu)

    # index, feature
    indexPath = os.path.join(params.model_dir, str(modelSlot.slotIndex), os.path.basename(modelSlot.indexFile))
    index = _loadIndex(indexPath)

    pipeline = Pipeline(
        embedder,
        inferencer,
        pitchExtractor,
        index,
        modelSlot.samplingRate,
        dev,
        half,
    )

    return pipeline


def _loadIndex(indexPath: str):
    # Indexのロード
    print("[Voice Changer] Loading index...")
    # ファイル指定があってもファイルがない場合はNone
    if os.path.exists(indexPath) is not True or os.path.isfile(indexPath) is not True:
        print("[Voice Changer] Index file is not found")
        return None

    try:
        print("Try loading...", indexPath)
        index = faiss.read_index(indexPath)
    except: # NOQA
        print("[Voice Changer] load index failed. Use no index.")
        traceback.print_exc()
        return None

    return index
