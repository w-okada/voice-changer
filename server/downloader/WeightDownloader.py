import os

from concurrent.futures import ThreadPoolExecutor
from downloader.Downloader import download
from mods.log_control import VoiceChangaerLogger
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
from settings import ServerSettings
from Exceptions import WeightsVerificationException
from utils.hasher import compute_hash
from xxhash import xxh128

logger = VoiceChangaerLogger.get_instance().getLogger()

def downloadWeight(voiceChangerParams: VoiceChangerParams | ServerSettings):
    content_vec_500_onnx = voiceChangerParams.content_vec_500_onnx
    # hubert_base = voiceChangerParams.hubert_base
    # hubert_base_jp = voiceChangerParams.hubert_base_jp
    # hubert_soft = voiceChangerParams.hubert_soft
    crepe_onnx_full = voiceChangerParams.crepe_onnx_full
    crepe_onnx_tiny = voiceChangerParams.crepe_onnx_tiny
    rmvpe = voiceChangerParams.rmvpe
    rmvpe_onnx = voiceChangerParams.rmvpe_onnx

    file_params = [
        # {
        #     "url": "https://huggingface.co/ddPn08/rvc-webui-models/resolve/main/embeddings/hubert_base.pt",
        #     "saveTo": hubert_base,
        #     "hash": "3ea71c977bf403eda3fcc73fb1bedc5a",
        # },
        # {
        #     "url": "https://huggingface.co/rinna/japanese-hubert-base/resolve/main/fairseq/model.pt",
        #     "saveTo": hubert_base_jp,
        #     "hash": "fed21bfb71a38df821cf9ae43e5da8b3",
        # },
        # {
        #     "url": "https://huggingface.co/wok000/weights/resolve/main/ddsp-svc30/embedder/hubert-soft-0d54a1f4.pt",
        #     "saveTo": hubert_soft,
        #     "hash": "c32e64b2d7e222a1912948a6192636d8",
        # },
        {
            "url": "https://huggingface.co/wok000/weights/resolve/main/crepe/onnx/full.onnx",
            "saveTo": crepe_onnx_full,
            "hash": "e9bb11eb5d3557805715077b30aefebc",
        },
        {
            "url": "https://huggingface.co/wok000/weights/resolve/main/crepe/onnx/tiny.onnx",
            "saveTo": crepe_onnx_tiny,
            "hash": "b509427f6d223152e57ff2aeb1b48300",
        },
        {
            "url": "https://huggingface.co/wok000/weights_gpl/resolve/main/content-vec/contentvec-f.onnx",
            "saveTo": content_vec_500_onnx,
            "hash": "ab288ca5b540a4a15909a40edf875d1e",
        },
        {
            "url": "https://huggingface.co/wok000/weights/resolve/main/rmvpe/rmvpe_20231006.pt",
            "saveTo": rmvpe,
            "hash": "7989809b6b54fb33653818e357bcb643",
        },
        {
            "url": "https://huggingface.co/wok000/weights_gpl/resolve/main/rmvpe/rmvpe_20231006.onnx",
            "saveTo": rmvpe_onnx,
            "hash": "b6979bf69503f8ec48c135000028a7b0",
        }
    ]

    files_to_download = []
    pos = 0
    for param in file_params:
        if os.path.exists(param["saveTo"]):
            continue
        files_to_download.append({
            "url": param["url"],
            "saveTo": param["saveTo"],
            "hash": param["hash"],
            "position": pos
        })
        pos += 1

    with ThreadPoolExecutor() as pool:
        pool.map(download, files_to_download)

    # ファイルサイズをログに書き込む。（デバッグ用）
    logger.info('[Voice Changer] Verifying files...')
    fail = False
    for param in file_params:
        file_path = param['saveTo']
        if not os.path.exists(file_path):
            fail = True
            continue

        hash = param["hash"]
        with open(file_path, 'rb') as f:
            received_hash = compute_hash(f, xxh128())
        if received_hash != hash:
            logger.error(f'Corrupted file {file_path}: calculated hash {received_hash}, expected hash {hash}')
            fail = True
        logger.debug(f"weight file [{file_path}]: {os.path.getsize(file_path)}")
    if fail:
        raise WeightsVerificationException()
    logger.info('[Voice Changer] Files were verified successfully!')
