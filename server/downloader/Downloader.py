import os
import json

from downloader.HttpClient import HttpClient
from tqdm import tqdm
from threading import Lock
from mods.log_control import VoiceChangaerLogger
from xxhash import xxh128
from utils.hasher import compute_hash
from const import ASSETS_FILE
from Exceptions import DownloadVerificationException

logger = VoiceChangaerLogger.get_instance().getLogger()

lock = Lock()

if os.path.exists(ASSETS_FILE):
    with open(ASSETS_FILE, encoding='utf-8') as f:
        files = json.load(f)
else:
    files = {}


async def download(params: dict):
    s = await HttpClient.get_client()

    url = params["url"]
    saveTo = params["saveTo"]
    position = params["position"]

    dirname = os.path.dirname(saveTo)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    try:
        offset = os.stat(saveTo).st_size
    except:
        offset = None

    expected_hash = params.get('hash')
    hasher = xxh128()
    if offset is not None:
        with open(saveTo, 'rb') as f:
            hash = compute_hash(f, hasher)
        # If hash was provided with the file - verify against provided hash
        if expected_hash is not None:
            if hash == expected_hash:
                logger.info(f'[Voice Changer] Verified {saveTo}')
                return
        # If hash was not provided - verify against local cache
        elif saveTo in files:
            fhash = files[saveTo]
            if hash == fhash:
                logger.info(f'[Voice Changer] Verified {saveTo}')
                return
    else:
        hash = None

    logger.info(f'[Voice Changer] Downloading {saveTo}...')
    res = await s.head(url, allow_redirects=True)
    res.raise_for_status()
    content_length = int(res.headers.get("content-length"))
    if offset is not None and offset == content_length:
        # Hash will not be written here if file is absent or incomplete
        write_file_entry(saveTo, hash)
        return
    accept_ranges = res.headers.get("Accept-Ranges")
    # If file is incomplete and web-server supports byte ranges - resume the download
    # In case file size somehow exceeds reported content-length - redownload the file completely
    resume_download = offset is not None and accept_ranges is not None and accept_ranges == 'bytes' and offset < content_length
    res = await s.get(
        url,
        headers={ 'Range': f'bytes={offset}-' } if resume_download else None,
        allow_redirects=True
    )
    res.raise_for_status()
    content_length = int(res.headers.get("content-length"))
    progress_bar = tqdm(
        total=content_length,
        leave=False,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        position=position,
    )

    # with tqdm
    with progress_bar, open(saveTo, "ab" if resume_download else "wb") as f:
        async for chunk in res.content.iter_chunked(8192):
            progress_bar.update(len(chunk))
            f.write(chunk)
            # Reusing the same hasher instance defined above
            hasher.update(chunk)

    # Get final hash (local chunks + remote chunks)
    hash = hasher.hexdigest()
    if expected_hash is not None and hash != expected_hash:
        raise DownloadVerificationException(saveTo, hash, expected_hash)

    write_file_entry(saveTo, hash)

def write_file_entry(saveTo: str, hash: str):
    global lock, files
    files[saveTo] = hash
    with lock, open(ASSETS_FILE, 'w', encoding='utf-8') as f:
        json.dump(files, f)
