from typing import Any, Protocol

from voice_changer.utils.VoiceChangerModel import AudioInOut, VoiceChangerModel


class VoiceChangerIF(Protocol):

    def get_processing_sampling_rate() -> int:
        ...

    def get_info(self) -> dict[str, Any]:
        ...

    def get_performance(self) -> list[int]:
        ...

    def setModel(model: VoiceChangerModel) -> None:
        ...

    def update_settings(self, key: str, val: int | float | str) -> bool:
        ...

    def on_request(receivedData: AudioInOut) -> tuple[AudioInOut, list[int | float]]:
        ...

    def export2onnx() -> Any:
        ...
