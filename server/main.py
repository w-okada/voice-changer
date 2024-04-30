import sys
import uvicorn
import asyncio
import traceback

from dotenv import set_key
from distutils.util import strtobool
from datetime import datetime
import platform
import os
import argparse
from downloader.WeightDownloader import downloadWeight
from downloader.SampleDownloader import downloadInitialSamples
from mods.ssl import create_self_signed_cert
from const import SSL_KEY_DIR

from settings import ServerSettings
from mods.log_control import VoiceChangaerLogger

VoiceChangaerLogger.get_instance().initialize(initialize=True)
logger = VoiceChangaerLogger.get_instance().getLogger()

def setupArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logLevel", type=str, default="error", help="Log level info|critical|error. (default: error)")
    parser.add_argument("--https", type=strtobool, default=False, help="use https")
    parser.add_argument("--httpsKey", type=str, default="ssl.key", help="path for the key of https")
    parser.add_argument("--httpsCert", type=str, default="ssl.cert", help="path for the cert of https")
    parser.add_argument("--httpsSelfSigned", type=strtobool, default=True, help="generate self-signed certificate")

    return parser

def printMessage(message, level=0):
    pf = platform.system()
    if pf == "Windows":
        if level == 0:
            message = f"{message}"
        elif level == 1:
            message = f"    {message}"
        elif level == 2:
            message = f"    {message}"
        else:
            message = f"    {message}"
    else:
        if level == 0:
            message = f"\033[17m{message}\033[0m"
        elif level == 1:
            message = f"\033[34m    {message}\033[0m"
        elif level == 2:
            message = f"\033[32m    {message}\033[0m"
        else:
            message = f"\033[47m    {message}\033[0m"
    logger.info(message)


async def runServer(host: str, port: int, logLevel: str = 'critical', key_path: str | None = None, cert_path: str | None = None):
    config = uvicorn.Config(
        "server:app_socketio",
        host=host,
        port=port,
        reload=False,
        ssl_keyfile=key_path,
        ssl_certfile=cert_path,
        log_level=logLevel
    )
    server = uvicorn.Server(config)
    await server.serve()


async def main(args):
    logger.debug(args)

    settings = ServerSettings()
    if not os.path.exists('.env'):
        for key, value in settings.model_dump().items():
            set_key('.env', key.upper(), str(value))

    printMessage(f"PYTHON: {sys.version}", level=2)
    # printMessage("Voice Changerを起動しています。", level=2)
    printMessage("Activating the Voice Changer.", level=2)
    # ダウンロード(Weight)

    await downloadWeight(settings)

    try:
        await downloadInitialSamples(settings.sample_mode, settings.model_dir)
    except Exception as e:
        print(traceback.format_exc())
        printMessage(f"Failed to download samples. Reason: {e}", level=2)

    # FIXME: Need to refactor samples download logic
    os.makedirs(settings.model_dir, exist_ok=True)

    # HTTPS key/cert作成
    if args.https and args.httpsSelfSigned:
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
        printMessage(f"protocol: HTTPS(self-signed), key:{key_path}, cert:{cert_path}", level=1)

    elif args.https and not args.httpsSelfSigned:
        # HTTPS
        key_path = args.httpsKey
        cert_path = args.httpsCert
        printMessage(f"protocol: HTTPS, key:{key_path}, cert:{cert_path}", level=1)
    else:
        # HTTP
        printMessage("protocol: HTTP", level=1)
    printMessage("-- ---- -- ", level=1)

    # アドレス表示
    # printMessage("ブラウザで次のURLを開いてください.", level=2)
    if args.https == 1:
        printMessage(f"The server is listening on https://{settings.host}:{settings.port}/", level=1)
    else:
        printMessage(f"The server is listening on http://{settings.host}:{settings.port}/", level=1)

    # サーバ起動
    if args.https:
        # HTTPS サーバ起動
        await runServer(settings.host, settings.port, args.logLevel, key_path, cert_path)
    else:
        await runServer(settings.host, settings.port, args.logLevel)


if __name__ == "__main__":
    parser = setupArgParser()
    args, _ = parser.parse_known_args()

    printMessage(f"Booting PHASE :{__name__}", level=2)

    try:
        asyncio.run(main(args))
    except Exception as e:
        print(traceback.format_exc())
        input('Press Enter to continue...')
