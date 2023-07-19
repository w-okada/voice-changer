

class VocoderOnnx:
    def __init__(self) -> None:
        pass

    def initialize(self, onnx_path: str, gpu: int):
        raise Exception("Not implemented")

    def infer(self, out_mel, pitch, silence_front, mask):
        raise Exception("Not implemented")
