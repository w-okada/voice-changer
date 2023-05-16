from dataclasses import dataclass


@dataclass
class VoiceChangerParams:
    model_dir: str
    samples: str
    content_vec_500: str
    content_vec_500_onnx: str
    content_vec_500_onnx_on: bool
    hubert_base: str
    hubert_base_jp: str
    hubert_soft: str
    nsf_hifigan: str
