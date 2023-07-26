import requests  # type: ignore
import os
from tqdm import tqdm

from mods.log_control import VoiceChangaerLogger

logger = VoiceChangaerLogger.get_instance().getLogger()


def download(params):
    url = params["url"]
    saveTo = params["saveTo"]
    position = params["position"]
    dirname = os.path.dirname(saveTo)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)

    try:
        req = requests.get(url, stream=True, allow_redirects=True)
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

    except Exception as e:
        logger.warning(e)


def download_no_tqdm(params):
    url = params["url"]
    saveTo = params["saveTo"]
    dirname = os.path.dirname(saveTo)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)
    try:
        req = requests.get(url, stream=True, allow_redirects=True)
        with open(saveTo, "wb") as f:
            countToDot = 0
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    countToDot += 1
                    if countToDot % 1024 == 0:
                        print(".", end="", flush=True)

        logger.info(f"[Voice Changer] download sample catalog. {saveTo}")
    except Exception as e:
        logger.warning(e)
