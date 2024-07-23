import asyncio

from downloader.Downloader import download
import logging
from settings import ServerSettings
from Exceptions import WeightDownloadException

logger = logging.getLogger(__name__)

async def downloadWeight(params: ServerSettings):
    logger.info('Loading weights.')
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
        {
            "url": "https://huggingface.co/wok000/weights/resolve/7a376af24f4f21f9d9e24160ed9d858e8a33bf93/crepe/onnx/full.onnx",
            "saveTo": params.crepe_onnx_full,
            "hash": "e9bb11eb5d3557805715077b30aefebc",
        },
        {
            "url": "https://huggingface.co/wok000/weights/resolve/7a376af24f4f21f9d9e24160ed9d858e8a33bf93/crepe/onnx/tiny.onnx",
            "saveTo": params.crepe_onnx_tiny,
            "hash": "b509427f6d223152e57ff2aeb1b48300",
        },
        {
            "url": "https://github.com/maxrmorrison/torchcrepe/raw/745670a18bf8c5f1a2f08c910c72433badde3e08/torchcrepe/assets/full.pth",
            "saveTo": params.crepe_full,
            "hash": "2ab425d128692f27ad5b765f13752333",
        },
        {
            "url": "https://github.com/maxrmorrison/torchcrepe/raw/745670a18bf8c5f1a2f08c910c72433badde3e08/torchcrepe/assets/tiny.pth",
            "saveTo": params.crepe_tiny,
            "hash": "eec11d7661587b6b90da7823cf409340",
        },
        {
            "url": "https://huggingface.co/wok000/weights_gpl/resolve/c2f3e4a8884dba0995347dfe24dc0ad40acb9eb7/content-vec/contentvec-f.onnx",
            "saveTo": params.content_vec_500_onnx,
            "hash": "ab288ca5b540a4a15909a40edf875d1e",
        },
        {
            "url": "https://huggingface.co/wok000/weights/resolve/4a9dbeb086b66721378b4fb29c84bf94d3e076ec/rmvpe/rmvpe_20231006.pt",
            "saveTo": params.rmvpe,
            "hash": "7989809b6b54fb33653818e357bcb643",
        },
        {
            "url": "https://huggingface.co/deiteris/weights/resolve/5040af391eb55d6415a209bfeb3089a866491670/rmvpe_upd.onnx",
            "saveTo": params.rmvpe_onnx,
            "hash": "9c6d7712f84d487ae781b0d7435c269b",
        },
        {
            "url": "https://github.com/CNChTu/FCPE/raw/819765c8db719c457f53aaee3238879ab98ed0cd/torchfcpe/assets/fcpe_c_v001.pt",
            "saveTo": params.fcpe,
            "hash": "933f1b588409b3945389381a2ab98014",
        },
        {
            "url": "https://huggingface.co/deiteris/weights/resolve/6abbb0285b1fc154e112b3c002ae63e1c1733d53/fcpe.onnx",
            "saveTo": params.fcpe_onnx,
            "hash": "6a7b11db05def00053102920d039760f",
        },
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
            logger.exception(res)
    if fail:
        raise WeightDownloadException()

    logger.info('All weights are loaded!')
