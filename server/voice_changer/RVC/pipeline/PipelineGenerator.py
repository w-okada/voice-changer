import os
import sys
import traceback
import faiss
import faiss.contrib.torch_utils
import torch
from Exceptions import PipelineCreateException
from data.ModelSlot import RVCModelSlot

from voice_changer.common.deviceManager.DeviceManager import DeviceManager
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
        inferencer = InferencerManager.getInferencer(modelSlot.modelType, modelPath, gpu, modelSlot.version)
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
    index, index_reconstruct = _loadIndex(indexPath, dev)

    pipeline = Pipeline(
        embedder,
        inferencer,
        pitchExtractor,
        index,
        index_reconstruct,
        modelSlot.samplingRate,
        modelSlot.embChannels,
        dev,
        half,
    )

    return pipeline


def _loadIndex(indexPath: str, dev: torch.device) -> tuple[faiss.Index | None, torch.Tensor | None]:
    # Indexのロード
    print("[Voice Changer] Loading index...")
    # ファイル指定があってもファイルがない場合はNone
    if os.path.exists(indexPath) is not True or os.path.isfile(indexPath) is not True:
        print("[Voice Changer] Index file is not found")
        return (None, None)

    try:
        print("Try loading...", indexPath)
        index: faiss.IndexIVFFlat = faiss.read_index(indexPath)
        if not index.is_trained:
            print("[Voice Changer] Invalid index. You MUST use added_xxxx.index, not trained_xxxx.index. Index will not be used.")
            return (None, None)
        # BUG: faiss-gpu does not support reconstruct on GPU indices
        # https://github.com/facebookresearch/faiss/issues/2181
        index_reconstruct = index.reconstruct_n(0, index.ntotal).to(dev)
        if sys.platform == 'linux' and dev.type == 'cuda':
            index: faiss.GpuIndexIVFFlat = faiss.index_cpu_to_gpus_list(index, gpus=[dev.index])
    except: # NOQA
        print("[Voice Changer] Load index failed. Use no index.")
        traceback.print_exc()
        return (None, None)

    return index, index_reconstruct
