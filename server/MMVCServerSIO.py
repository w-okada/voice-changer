import sys

from distutils.util import strtobool
from datetime import datetime
import socket
import platform
import os
import argparse
from Exceptions import WeightDownladException
from downloader.SampleDownloader import downloadInitialSamples
from downloader.WeightDownloader import downloadWeight
from voice_changer.VoiceChangerParamsManager import VoiceChangerParamsManager

from voice_changer.utils.VoiceChangerParams import VoiceChangerParams

import uvicorn
from mods.ssl import create_self_signed_cert
from voice_changer.VoiceChangerManager import VoiceChangerManager
from sio.MMVC_SocketIOApp import MMVC_SocketIOApp
from restapi.MMVC_Rest import MMVC_Rest
from const import (
    NATIVE_CLIENT_FILE_MAC,
    NATIVE_CLIENT_FILE_WIN,
    SSL_KEY_DIR,
)
import subprocess
import multiprocessing as mp

from mods.log_control import VoiceChangaerLogger


if __name__ == "__main__":
    VoiceChangaerLogger.get_instance().initialize(initialize=True)
else:
    VoiceChangaerLogger.get_instance().initialize(initialize=False)


logger = VoiceChangaerLogger.get_instance().getLogger()
logger.debug(f"---------------- Booting PHASE :{__name__} -----------------")


def setupArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logLevel", type=str, default="error", help="Log level info|critical|error. (default: error)")
    parser.add_argument("-p", type=int, default=18888, help="port")
    parser.add_argument("--https", type=strtobool, default=False, help="use https")
    parser.add_argument("--test_connect", type=str, default="8.8.8.8", help="test connect to detect ip in https mode. default 8.8.8.8")
    parser.add_argument("--httpsKey", type=str, default="ssl.key", help="path for the key of https")
    parser.add_argument("--httpsCert", type=str, default="ssl.cert", help="path for the cert of https")
    parser.add_argument("--httpsSelfSigned", type=strtobool, default=True, help="generate self-signed certificate")

    parser.add_argument("--model_dir", type=str, default="model_dir", help="path to model files")
    parser.add_argument("--sample_mode", type=str, default="production", help="rvc_sample_mode")

    parser.add_argument("--content_vec_500", type=str, default="pretrain/checkpoint_best_legacy_500.pt", help="path to content_vec_500 model(pytorch)")
    parser.add_argument("--content_vec_500_onnx", type=str, default="pretrain/content_vec_500.onnx", help="path to content_vec_500 model(onnx)")
    parser.add_argument("--content_vec_500_onnx_on", type=strtobool, default=True, help="use or not onnx for  content_vec_500")
    parser.add_argument("--hubert_base", type=str, default="pretrain/hubert_base.pt", help="path to hubert_base model(pytorch)")
    parser.add_argument("--hubert_base_jp", type=str, default="pretrain/rinna_hubert_base_jp.pt", help="path to hubert_base_jp model(pytorch)")
    parser.add_argument("--hubert_soft", type=str, default="pretrain/hubert/hubert-soft-0d54a1f4.pt", help="path to hubert_soft model(pytorch)")
    parser.add_argument("--nsf_hifigan", type=str, default="pretrain/nsf_hifigan/model", help="path to nsf_hifigan model(pytorch)")
    parser.add_argument("--crepe_onnx_full", type=str, default="pretrain/crepe_onnx_full.onnx", help="path to crepe_onnx_full")
    parser.add_argument("--crepe_onnx_tiny", type=str, default="pretrain/crepe_onnx_tiny.onnx", help="path to crepe_onnx_tiny")
    parser.add_argument("--rmvpe", type=str, default="pretrain/rmvpe.pt", help="path to rmvpe")
    parser.add_argument("--rmvpe_onnx", type=str, default="pretrain/rmvpe.onnx", help="path to rmvpe onnx")

    parser.add_argument("--host", type=str, default='127.0.0.1', help="IP address of the network interface to listen for HTTP connections. Specify 0.0.0.0 to listen on all interfaces.")
    parser.add_argument("--allowed-origins", action='append', default=[], help="List of URLs to allow connection from, i.e. https://example.com. Allows http(s)://127.0.0.1:{port} and http(s)://localhost:{port} by default.")

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


parser = setupArgParser()
args, unknown = parser.parse_known_args()
voiceChangerParams = VoiceChangerParams(
    model_dir=args.model_dir,
    content_vec_500=args.content_vec_500,
    content_vec_500_onnx=args.content_vec_500_onnx,
    content_vec_500_onnx_on=args.content_vec_500_onnx_on,
    hubert_base=args.hubert_base,
    hubert_base_jp=args.hubert_base_jp,
    hubert_soft=args.hubert_soft,
    nsf_hifigan=args.nsf_hifigan,
    crepe_onnx_full=args.crepe_onnx_full,
    crepe_onnx_tiny=args.crepe_onnx_tiny,
    rmvpe=args.rmvpe,
    rmvpe_onnx=args.rmvpe_onnx,
    sample_mode=args.sample_mode,
)
vcparams = VoiceChangerParamsManager.get_instance()
vcparams.setParams(voiceChangerParams)

printMessage(f"Booting PHASE :{__name__}", level=2)

HOST = args.host
PORT = args.p


def localServer(logLevel: str = 'critical', key_path: str | None = None, cert_path: str | None = None):
    try:
        uvicorn.run(
            f"{os.path.basename(__file__)[:-3]}:app_socketio",
            host=HOST,
            port=int(PORT),
            reload=False,
            ssl_keyfile=key_path,
            ssl_certfile=cert_path,
            log_level=logLevel,
        )
    except Exception as e:
        logger.error(f"[Voice Changer] Web Server Launch Exception, {e}")


if __name__ == "MMVCServerSIO":
    mp.freeze_support()

    voiceChangerManager = VoiceChangerManager.get_instance(voiceChangerParams)
    app_fastapi = MMVC_Rest.get_instance(voiceChangerManager, voiceChangerParams, args.allowed_origins, PORT)
    app_socketio = MMVC_SocketIOApp.get_instance(app_fastapi, voiceChangerManager, args.allowed_origins, PORT)


if __name__ == "__mp_main__":
    # printMessage("サーバプロセスを起動しています。", level=2)
    printMessage("The server process is starting up.", level=2)

if __name__ == "__main__":
    mp.freeze_support()

    logger.debug(args)

    printMessage(f"PYTHON:{sys.version}", level=2)
    # printMessage("Voice Changerを起動しています。", level=2)
    printMessage("Activating the Voice Changer.", level=2)
    # ダウンロード(Weight)
    try:
        downloadWeight(voiceChangerParams)
    except WeightDownladException:
        # printMessage("RVC用のモデルファイルのダウンロードに失敗しました。", level=2)
        printMessage("failed to download weight for rvc", level=2)

    # ダウンロード(Sample)
    try:
        downloadInitialSamples(args.sample_mode, args.model_dir)
    except Exception as e:
        printMessage(f"[Voice Changer] loading sample failed {e}", level=2)

    # PORT = args.p

    if os.getenv("EX_PORT"):
        EX_PORT = os.environ["EX_PORT"]
        printMessage(f"External_Port:{EX_PORT} Internal_Port:{PORT}", level=1)
    else:
        printMessage(f"Internal_Port:{PORT}", level=1)

    if os.getenv("EX_IP"):
        EX_IP = os.environ["EX_IP"]
        printMessage(f"External_IP:{EX_IP}", level=1)

    # HTTPS key/cert作成
    if args.https and args.httpsSelfSigned == 1:
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

    elif args.https and args.httpsSelfSigned == 0:
        # HTTPS
        key_path = args.httpsKey
        cert_path = args.httpsCert
        printMessage(f"protocol: HTTPS, key:{key_path}, cert:{cert_path}", level=1)
    else:
        # HTTP
        printMessage("protocol: HTTP", level=1)
    printMessage("-- ---- -- ", level=1)

    # アドレス表示
    printMessage("Please open the following URL in your browser.", level=2)
    # printMessage("ブラウザで次のURLを開いてください.", level=2)
    if args.https == 1:
        printMessage("https://<IP>:<PORT>/", level=1)
    else:
        printMessage("http://<IP>:<PORT>/", level=1)

    # printMessage("多くの場合は次のいずれかのURLにアクセスすると起動します。", level=2)
    printMessage("In many cases, it will launch when you access any of the following URLs.", level=2)
    if "EX_PORT" in locals() and "EX_IP" in locals():  # シェルスクリプト経由起動(docker)
        if args.https == 1:
            printMessage(f"https://localhost:{EX_PORT}/", level=1)
            for ip in EX_IP.strip().split(" "):
                printMessage(f"https://{ip}:{EX_PORT}/", level=1)
        else:
            printMessage(f"http://localhost:{EX_PORT}/", level=1)
    else:  # 直接python起動
        if args.https == 1:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((args.test_connect, 80))
            hostname = s.getsockname()[0]
            printMessage(f"https://localhost:{PORT}/", level=1)
            printMessage(f"https://{hostname}:{PORT}/", level=1)
        else:
            printMessage(f"http://localhost:{PORT}/", level=1)

    # サーバ起動
    if args.https:
        # HTTPS サーバ起動
        try:
            localServer(args.logLevel, key_path, cert_path)
        except Exception as e:
            logger.error(f"[Voice Changer] Web Server(https) Launch Exception, {e}")

    else:
        p = mp.Process(name="p", target=localServer, args=(args.logLevel,))
        p.start()
        try:
            if sys.platform.startswith("win"):
                process = subprocess.Popen([NATIVE_CLIENT_FILE_WIN, "--disable-gpu", "-u", f"http://localhost:{PORT}/"])
                return_code = process.wait()
                logger.info("client closed.")
                p.terminate()
            elif sys.platform.startswith("darwin"):
                process = subprocess.Popen([NATIVE_CLIENT_FILE_MAC, "--disable-gpu", "-u", f"http://localhost:{PORT}/"])
                return_code = process.wait()
                logger.info("client closed.")
                p.terminate()

        except Exception as e:
            logger.error(f"[Voice Changer] Client Launch Exception, {e}")
