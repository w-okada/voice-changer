import os
import multiprocessing as mp
# NOTE: This is required to avoid recursive process call bug for macOS
mp.freeze_support()
from const import SSL_KEY_DIR, DOTENV_FILE, ROOT_PATH, UPLOAD_DIR, TMP_DIR, LOG_FILE, get_version, get_edition
# NOTE: This is required to fix current working directory on macOS
os.chdir(ROOT_PATH)

import sys
import uvicorn
import asyncio

import threading
import socket
import time
import logging
from dotenv import set_key
from utils.strtobool import strtobool
from datetime import datetime
import argparse
from downloader.WeightDownloader import downloadWeight
from downloader.SampleDownloader import downloadInitialSamples
from mods.ssl import create_self_signed_cert
from webbrowser import open_new_tab
from settings import ServerSettings

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(levelname)-8s [%(module)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), stream_handler]
)
logger = logging.getLogger(__name__)
settings = ServerSettings()

def setupArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default="error", help="Log level info|critical|error.")
    parser.add_argument("--https", type=strtobool, default=False, help="use https")
    parser.add_argument("--https-key", type=str, default="ssl.key", help="path for the key of https")
    parser.add_argument("--https-cert", type=str, default="ssl.cert", help="path for the cert of https")
    parser.add_argument("--https-self-signed", type=strtobool, default=True, help="generate self-signed certificate")

    return parser

def check_port(port) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))

def wait_for_server(proto: str, launch_browser: bool):
    while True:
        time.sleep(1)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex(('127.0.0.1', settings.port))
        if result == 0:
            break
    logger.info('-' * 8)
    logger.info(f"The server is listening on {proto}://{settings.host}:{settings.port}/")
    logger.info('-' * 8)
    if launch_browser:
        open_new_tab(f'{proto}://127.0.0.1:{settings.port}')

async def runServer(host: str, port: int, launch_browser: bool = False, log_level: str = 'error', key_path: str | None = None, cert_path: str | None = None):
    check_port(port)

    config = uvicorn.Config(
        "app:socketio",
        host=host,
        port=port,
        reload=False,
        ssl_keyfile=key_path,
        ssl_certfile=cert_path,
        log_level=log_level
    )
    server = uvicorn.Server(config)

    proto = 'https' if key_path and cert_path else 'http'
    threading.Thread(target=wait_for_server, daemon=True, args=(proto, launch_browser)).start()

    await server.serve()

async def main(args):
    logger.debug(args)

    logger.info(f"Python: {sys.version}")
    logger.info(f"Voice changer version: {get_version()} {get_edition()}")
    # ダウンロード(Weight)

    await downloadWeight(settings)

    try:
        await downloadInitialSamples(settings.sample_mode, settings.model_dir)
    except Exception as e:
        logger.error(f"Failed to download samples.")
        logger.exception(e)

    # FIXME: Need to refactor samples download logic
    os.makedirs(settings.model_dir, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    # HTTPS key/cert作成
    if args.https and args.https_self_signed:
        # HTTPS(おれおれ証明書生成)
        os.makedirs(SSL_KEY_DIR, exist_ok=True)
        key_base_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        keyname = f"{key_base_name}.key"
        certname = f"{key_base_name}.cert"
        create_self_signed_cert(
            certname,
            keyname,
            certargs={
                "Country": "JP",
                "State": "Tokyo",
                "City": "Chuo-ku",
                "Organization": "F",
                "Org. Unit": "F",
            },
            cert_dir=SSL_KEY_DIR,
        )
        key_path = os.path.join(SSL_KEY_DIR, keyname)
        cert_path = os.path.join(SSL_KEY_DIR, certname)
        logger.info(f"protocol: HTTPS(self-signed), key:{key_path}, cert:{cert_path}")

    elif args.https and not args.https_self_signed:
        # HTTPS
        key_path = args.https_key
        cert_path = args.https_cert
        logger.info(f"protocol: HTTPS, key:{key_path}, cert:{cert_path}")
    else:
        # HTTP
        logger.info("protocol: HTTP")

    # サーバ起動
    if args.https:
        # HTTPS サーバ起動
        await runServer(settings.host, settings.port, args.launch_browser, args.log_level, key_path, cert_path)
    else:
        await runServer(settings.host, settings.port, args.launch_browser, args.log_level)


if __name__ == "__main__":
    parser = setupArgParser()
    args, _ = parser.parse_known_args()
    args.launch_browser = False

    try:
        asyncio.run(main(args))
    except Exception as e:
        logger.exception(e)
        input('Press Enter to continue...')
