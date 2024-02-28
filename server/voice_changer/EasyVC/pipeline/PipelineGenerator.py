import os
import traceback
from Exceptions import PipelineCreateException
from const import EnumInferenceTypes, PitchExtractorType
from data.ModelSlot import EasyVCModelSlot
from voice_changer.EasyVC.pipeline.Pipeline import Pipeline

from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.embedder.EmbedderManager import EmbedderManager
from voice_changer.RVC.inferencer.InferencerManager import InferencerManager
from voice_changer.RVC.pitchExtractor.PitchExtractorManager import PitchExtractorManager
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams


def createPipeline(params: VoiceChangerParams, modelSlot: EasyVCModelSlot, gpu: int, f0Detector: PitchExtractorType):
    dev = DeviceManager.get_instance().getDevice(gpu)
    half = DeviceManager.get_instance().halfPrecisionAvailable(gpu)

    # Inferencer 生成
    try:
        modelPath = os.path.join(params.model_dir, str(modelSlot.slotIndex), os.path.basename(modelSlot.modelFile))
        inferencer = InferencerManager.getInferencer(EnumInferenceTypes.easyVC, modelPath, gpu, modelSlot.version)
    except Exception as e:
        print("[Voice Changer] exception! loading inferencer", e)
        traceback.print_exc()
        raise PipelineCreateException("[Voice Changer] exception! loading inferencer")

    # Embedder 生成
    try:
        embedder = EmbedderManager.getEmbedder(
            "whisper",
            half,
            dev,
        )
    except Exception as e:
        print("[Voice Changer] exception! loading embedder", e, dev)
        traceback.print_exc()
        raise PipelineCreateException("[Voice Changer] exception! loading embedder")

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
