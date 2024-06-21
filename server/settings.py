import os

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from const import EDITION_FILE, RVCSampleMode, ROOT_PATH, DOTENV_FILE

class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=DOTENV_FILE, env_file_encoding='utf-8', protected_namespaces=('model_config',))

    model_dir: str = 'model_dir'
    content_vec_500: str = 'pretrain/checkpoint_best_legacy_500.pt'
    content_vec_500_onnx: str = 'pretrain/content_vec_500.onnx'
    content_vec_500_onnx_on: bool = True
    # hubert_base: str = 'pretrain/hubert_base.pt'
    # hubert_base_jp: str = 'pretrain/rinna_hubert_base_jp.pt'
    # hubert_soft: str = 'pretrain/hubert/hubert-soft-0d54a1f4.pt'
    sample_mode: RVCSampleMode = ''
    crepe_onnx_full: str = 'pretrain/crepe_onnx_full.onnx'
    crepe_onnx_tiny: str = 'pretrain/crepe_onnx_tiny.onnx'
    rmvpe: str = 'pretrain/rmvpe.pt'
    rmvpe_onnx: str = 'pretrain/rmvpe.onnx'
    fcpe: str = 'pretrain/fcpe.pt'
    fcpe_onnx: str = 'pretrain/fcpe.onnx'
    host: str = '127.0.0.1'
    port: int = 18888
    allowed_origins: Literal['*'] | list[str] = []
    edition: str = open(EDITION_FILE, 'r').read()

def resolve_paths(settings: ServerSettings) -> ServerSettings:
    settings.model_dir = os.path.join(ROOT_PATH, settings.model_dir)
    settings.content_vec_500_onnx = os.path.join(ROOT_PATH, settings.content_vec_500_onnx)
    settings.crepe_onnx_full = os.path.join(ROOT_PATH, settings.crepe_onnx_full)
    settings.crepe_onnx_tiny = os.path.join(ROOT_PATH, settings.crepe_onnx_tiny)
    settings.rmvpe = os.path.join(ROOT_PATH, settings.rmvpe)
    settings.rmvpe_onnx = os.path.join(ROOT_PATH, settings.rmvpe_onnx)
    settings.fcpe = os.path.join(ROOT_PATH, settings.fcpe)
    settings.fcpe_onnx = os.path.join(ROOT_PATH, settings.fcpe_onnx)
    return settings
