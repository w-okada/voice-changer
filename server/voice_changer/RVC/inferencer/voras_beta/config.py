from typing import *

from pydantic import BaseModel


class TrainConfigTrain(BaseModel):
    log_interval: int
    seed: int
    epochs: int
    learning_rate: float
    betas: List[float]
    eps: float
    batch_size: int
    fp16_run: bool
    lr_decay: float
    segment_size: int
    init_lr_ratio: int
    warmup_epochs: int
    c_mel: int
    c_kl: float


class TrainConfigData(BaseModel):
    max_wav_value: float
    sampling_rate: int
    filter_length: int
    hop_length: int
    win_length: int
    n_mel_channels: int
    mel_fmin: float
    mel_fmax: Any


class TrainConfigModel(BaseModel):
    emb_channels: int
    inter_channels: int
    n_layers: int
    upsample_rates: List[int]
    use_spectral_norm: bool
    gin_channels: int
    spk_embed_dim: int


class TrainConfig(BaseModel):
    version: Literal["voras"] = "voras"
    train: TrainConfigTrain
    data: TrainConfigData
    model: TrainConfigModel


class DatasetMetaItem(BaseModel):
    gt_wav: str
    co256: str
    f0: Optional[str]
    f0nsf: Optional[str]
    speaker_id: int


class DatasetMetadata(BaseModel):
    files: Dict[str, DatasetMetaItem]
    # mute: DatasetMetaItem
