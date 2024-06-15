from typing import Protocol

from voice_changer.utils.LoadModelParams import LoadModelParams


class ModelSlotGenerator(Protocol):
    @classmethod
    def load_model(cls, params: LoadModelParams):
        ...
