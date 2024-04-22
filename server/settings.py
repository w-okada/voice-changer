from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from const import EDITION_FILE

class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', protected_namespaces=('model_config',))

    model_dir: str = 'model_dir'
    content_vec_500: str = ''
    content_vec_500_onnx: str = 'pretrain/content_vec_500.onnx'
    content_vec_500_onnx_on: bool = True
    hubert_base: str = 'pretrain/hubert_base.pt'
    hubert_base_jp: str = 'pretrain/rinna_hubert_base_jp.pt'
    hubert_soft: str = 'pretrain/hubert/hubert-soft-0d54a1f4.pt'
    nsf_hifigan: str = 'pretrain/nsf_hifigan/model'
    sample_mode: str = 'production'
    crepe_onnx_full: str = 'pretrain/crepe_onnx_full.onnx'
    crepe_onnx_tiny: str = 'pretrain/crepe_onnx_tiny.onnx'
    rmvpe: str = 'pretrain/rmvpe.pt'
    rmvpe_onnx: str = 'pretrain/rmvpe.onnx'
    host: str = '127.0.0.1'
    port: int = 18888
    allowed_origins: Literal['*'] | list[str] = []
    edition: str = open(EDITION_FILE, 'r').read()