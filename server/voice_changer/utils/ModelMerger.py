from typing import Protocol
from const import VoiceChangerType
from dataclasses import dataclass


@dataclass
class MergeElement:
    slotIndex: int
    strength: int


@dataclass
class ModelMergerRequest:
    voiceChangerType: VoiceChangerType
    command: str
    files: list[MergeElement]


class ModelMerger(Protocol):
    @classmethod
    def merge_models(cls, request: ModelMergerRequest):
        ...
