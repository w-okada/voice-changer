import os

from downloader.HttpClient import HttpClient
from tqdm import tqdm

from mods.log_control import VoiceChangaerLogger

logger = VoiceChangaerLogger.get_instance().getLogger()


async def download(params):
    s = await HttpClient.get_client()

    url = params["url"]
    saveTo = params["saveTo"]
    position = params["position"]
    offset = params["offset"]
    dirname = os.path.dirname(saveTo)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)

    res = await s.head(url, allow_redirects=True)
    ranges = res.headers.get("Accept-Ranges")
    # If file is incomplete and web-server supports byte ranges - resume download
    resume_download = offset is not None and ranges is not None and 'bytes' in ranges
    if resume_download:
        res = await s.get(url, headers={ 'Range': f'bytes={offset}-' }, allow_redirects=True)
    else:
        res = await s.get(url, allow_redirects=True)
    content_length = res.headers.get("content-length")
    progress_bar = tqdm(
        total=int(content_length) if content_length is not None else None,
        leave=False,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        position=position,
    )

    # with tqdm
    with open(saveTo, "ab" if resume_download else "wb") as f:
        async for chunk in res.content.iter_chunked(4096):
            progress_bar.update(len(chunk))
            f.write(chunk)


async def download_no_tqdm(params):
    s = await HttpClient.get_client()

    url = params["url"]
    saveTo = params["saveTo"]
    dirname = os.path.dirname(saveTo)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)

    res = await s.get(url, allow_redirects=True)
    with open(saveTo, "wb") as f:
        countToDot = 0
        async for chunk in res.content.iter_chunked(4096):
            f.write(chunk)
            countToDot += 1
            if countToDot % 4096 == 0:
                print(".", end="", flush=True)
