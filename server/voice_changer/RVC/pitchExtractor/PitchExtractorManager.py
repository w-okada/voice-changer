from typing import Protocol
from const import PitchExtractorType
from voice_changer.RVC.pitchExtractor.CrepeOnnxPitchExtractor import CrepeOnnxPitchExtractor
from voice_changer.RVC.pitchExtractor.DioPitchExtractor import DioPitchExtractor
from voice_changer.RVC.pitchExtractor.HarvestPitchExtractor import HarvestPitchExtractor
from voice_changer.RVC.pitchExtractor.CrepePitchExtractor import CrepePitchExtractor
from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.RVC.pitchExtractor.RMVPEOnnxPitchExtractor import RMVPEOnnxPitchExtractor
from voice_changer.RVC.pitchExtractor.RMVPEPitchExtractor import RMVPEPitchExtractor
from settings import ServerSettings


class PitchExtractorManager(Protocol):
    currentPitchExtractor: PitchExtractor | None = None
    params: ServerSettings
    gpu: int = -1

    @classmethod
    def initialize(cls, params: ServerSettings):
        cls.params = params

    @classmethod
    def getPitchExtractor(
        cls, pitchExtractorType: PitchExtractorType, gpu: int
    ) -> PitchExtractor:
        cls.currentPitchExtractor = cls.loadPitchExtractor(pitchExtractorType, gpu)
        return cls.currentPitchExtractor

    @classmethod
    def loadPitchExtractor(
        cls, pitchExtractorType: PitchExtractorType, gpu: int
    ) -> PitchExtractor:
        if cls.currentPitchExtractor is not None \
            and pitchExtractorType == cls.currentPitchExtractor.pitchExtractorType and gpu == cls.gpu:
            print('[Voice Changer] Reusing pitch extractor.')
            return cls.currentPitchExtractor

        cls.gpu = gpu
        print(f'[Voice Changer] Loading pitch extractor {pitchExtractorType}')
        try:
            if pitchExtractorType == "harvest":
                return HarvestPitchExtractor()
            elif pitchExtractorType == "dio":
                return DioPitchExtractor()
            elif pitchExtractorType == "crepe_tiny":
                return CrepePitchExtractor(gpu, 'tiny')
            elif pitchExtractorType == "crepe_full":
                return CrepePitchExtractor(gpu, 'full')
            elif pitchExtractorType == "crepe_tiny_onnx":
                return CrepeOnnxPitchExtractor(pitchExtractorType, cls.params.crepe_onnx_tiny, gpu)
            elif pitchExtractorType == "crepe_full_onnx":
                return CrepeOnnxPitchExtractor(pitchExtractorType, cls.params.crepe_onnx_full, gpu)
            elif pitchExtractorType == "rmvpe":
                return RMVPEPitchExtractor(cls.params.rmvpe, gpu)
            elif pitchExtractorType == "rmvpe_onnx":
                return RMVPEOnnxPitchExtractor(cls.params.rmvpe_onnx, gpu)
            else:
                # return hubert as default
                print(f"[Voice Changer] PitchExctractor not found {pitchExtractorType}. Fallback to rmvpe_onnx")
                return RMVPEOnnxPitchExtractor(cls.params.rmvpe_onnx, gpu)
        except RuntimeError as e:
            import traceback
            print(traceback.format_exc())
            print(e)
            print(f'[Voice Changer] Failed to load {pitchExtractorType}. Fallback to rmvpe_onnx.')
            return RMVPEOnnxPitchExtractor(cls.params.rmvpe_onnx, gpu)
