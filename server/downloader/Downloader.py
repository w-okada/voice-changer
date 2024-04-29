import os

from downloader.HttpClient import HttpClient
from tqdm import tqdm

from mods.log_control import VoiceChangaerLogger

logger = VoiceChangaerLogger.get_instance().getLogger()


def download(params):
    s = HttpClient.get_client()

    url = params["url"]
    saveTo = params["saveTo"]
    position = params["position"]
    dirname = os.path.dirname(saveTo)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)

    req = s.get(url, stream=True, allow_redirects=True)
    content_length = req.headers.get("content-length")
    progress_bar = tqdm(
        total=int(content_length) if content_length is not None else None,
        leave=False,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        position=position,
    )

    # with tqdm
    with open(saveTo, "wb") as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                progress_bar.update(len(chunk))
                f.write(chunk)


def download_no_tqdm(params):
    s = HttpClient.get_client()

    url = params["url"]
    saveTo = params["saveTo"]
    dirname = os.path.dirname(saveTo)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)

    req = s.get(url, stream=True, allow_redirects=True)
    with open(saveTo, "wb") as f:
        countToDot = 0
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                countToDot += 1
                if countToDot % 1024 == 0:
                    print(".", end="", flush=True)
