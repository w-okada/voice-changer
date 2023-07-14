import traceback
from data.ModelSlot import DiffusionSVCModelSlot
from voice_changer.DiffusionSVC.inferencer.InferencerManager import InferencerManager
from voice_changer.DiffusionSVC.pipeline.Pipeline import Pipeline
from voice_changer.DiffusionSVC.pitchExtractor.PitchExtractorManager import PitchExtractorManager

from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.embedder.EmbedderManager import EmbedderManager


def createPipeline(modelSlot: DiffusionSVCModelSlot, gpu: int, f0Detector: str):
    dev = DeviceManager.get_instance().getDevice(gpu)
    # half = DeviceManager.get_instance().halfPrecisionAvailable(gpu)
    half = False

    # Inferencer 生成
    try:
        inferencer = InferencerManager.getInferencer(modelSlot.modelType, modelSlot.modelFile, gpu)
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
    pitchExtractor = PitchExtractorManager.getPitchExtractor(f0Detector, gpu)

    pipeline = Pipeline(
        embedder,
        inferencer,
        pitchExtractor,
        modelSlot.samplingRate,
        dev,
        half,
    )

    return pipeline

