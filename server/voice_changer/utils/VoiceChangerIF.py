from typing import Any, Protocol

from voice_changer.utils.VoiceChangerModel import AudioInOut, VoiceChangerModel


class VoiceChangerIF(Protocol):

    def get_processing_sampling_rate() -> int:
        ...

    def get_info(self) -> dict[str, Any]:
        ...

    def set_model(model: VoiceChangerModel) -> None:
        ...

    def update_settings(self, key: str, val: Any, old_val: Any):
        ...

    def on_request(receivedData: AudioInOut) -> tuple[AudioInOut, list[int | float]]:
        ...

    def export2onnx() -> Any:
        ...

    def set_input_sample_rate(self):
        ...

    def set_output_sample_rate(self):
        ...
