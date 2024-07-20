from const import EmbedderType
from voice_changer.RVC.embedder.Embedder import Embedder
from voice_changer.RVC.embedder.OnnxContentvec import OnnxContentvec
from settings import ServerSettings
import logging
logger = logging.getLogger(__name__)

class EmbedderManager:
    embedder: Embedder | None = None
    params: ServerSettings

    @classmethod
    def initialize(cls, params: ServerSettings):
        cls.params = params

    @classmethod
    def get_embedder(cls, embedder_type: EmbedderType, force_reload: bool = False) -> Embedder:
        if cls.embedder is not None \
            and cls.embedder.matchCondition(embedder_type) \
            and not force_reload:
            logger.info('Reusing embedder.')
            return cls.embedder
        cls.embedder = cls.load_embedder(embedder_type)
        return cls.embedder

    @classmethod
    def load_embedder(cls, embedder_type: EmbedderType) -> Embedder:
        if embedder_type not in ["hubert_base", "contentvec"]:
            raise RuntimeError(f'Unsupported embedder type: {embedder_type}')
        file = cls.params.content_vec_500_onnx
        return OnnxContentvec().load_model(file)
