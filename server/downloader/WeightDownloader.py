import asyncio

from downloader.Downloader import download
from mods.log_control import VoiceChangaerLogger
from settings import ServerSettings
from Exceptions import WeightDownloadException

logger = VoiceChangaerLogger.get_instance().getLogger()

async def downloadWeight(params: ServerSettings):
    logger.info('[Voice Changer] Loading weights.')
    file_params = [
        # {
        #     "url": "https://huggingface.co/ddPn08/rvc-webui-models/resolve/main/embeddings/hubert_base.pt",
        #     "saveTo": params.hubert_base,
        #     "hash": "3ea71c977bf403eda3fcc73fb1bedc5a",
        # },
        # {
        #     "url": "https://huggingface.co/rinna/japanese-hubert-base/resolve/main/fairseq/model.pt",
        #     "saveTo": params.hubert_base_jp,
        #     "hash": "fed21bfb71a38df821cf9ae43e5da8b3",
        # },
        # {
        #     "url": "https://huggingface.co/wok000/weights/resolve/main/ddsp-svc30/embedder/hubert-soft-0d54a1f4.pt",
        #     "saveTo": params.hubert_soft,
        #     "hash": "c32e64b2d7e222a1912948a6192636d8",
        # },
        {
            "url": "https://huggingface.co/wok000/weights/resolve/main/crepe/onnx/full.onnx",
            "saveTo": params.crepe_onnx_full,
            "hash": "e9bb11eb5d3557805715077b30aefebc",
        },
        {
            "url": "https://huggingface.co/wok000/weights/resolve/main/crepe/onnx/tiny.onnx",
            "saveTo": params.crepe_onnx_tiny,
            "hash": "b509427f6d223152e57ff2aeb1b48300",
        },
        {
            "url": "https://huggingface.co/wok000/weights_gpl/resolve/main/content-vec/contentvec-f.onnx",
            "saveTo": params.content_vec_500_onnx,
            "hash": "ab288ca5b540a4a15909a40edf875d1e",
        },
        {
            "url": "https://huggingface.co/wok000/weights/resolve/main/rmvpe/rmvpe_20231006.pt",
            "saveTo": params.rmvpe,
            "hash": "7989809b6b54fb33653818e357bcb643",
        },
        {
            "url": "https://huggingface.co/wok000/weights_gpl/resolve/main/rmvpe/rmvpe_20231006.onnx",
            "saveTo": params.rmvpe_onnx,
            "hash": "b6979bf69503f8ec48c135000028a7b0",
        }
    ]

    files_to_download = []
    for param in file_params:
        files_to_download.append({
            "url": param["url"],
            "saveTo": param['saveTo'],
            "hash": param['hash'],
        })

    tasks: list[asyncio.Task] = []
    for file in files_to_download:
        tasks.append(asyncio.ensure_future(download(file)))
    fail = False
    for res in await asyncio.gather(*tasks, return_exceptions=True):
        if isinstance(res, Exception):
            fail = True
            logger.error(f'[Voice Changer] {res}')
    if fail:
        raise WeightDownloadException()

    logger.info('[Voice Changer] All weights are loaded!')
