from concurrent.futures import ThreadPoolExecutor
import sys

from distutils.util import strtobool
from datetime import datetime
import socket
import platform
import os
import argparse
import requests  # type: ignore

from tqdm import tqdm
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams

import uvicorn
from mods.ssl import create_self_signed_cert
from voice_changer.VoiceChangerManager import VoiceChangerManager
from sio.MMVC_SocketIOApp import MMVC_SocketIOApp
from restapi.MMVC_Rest import MMVC_Rest
from const import NATIVE_CLIENT_FILE_MAC, NATIVE_CLIENT_FILE_WIN, SSL_KEY_DIR
import subprocess
import multiprocessing as mp
from misc.log_control import setup_loggers

setup_loggers()


def setupArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=int, default=18888, help="port")
    parser.add_argument("--https", type=strtobool, default=False, help="use https")
    parser.add_argument(
        "--httpsKey", type=str, default="ssl.key", help="path for the key of https"
    )
    parser.add_argument(
        "--httpsCert", type=str, default="ssl.cert", help="path for the cert of https"
    )
    parser.add_argument(
        "--httpsSelfSigned",
        type=strtobool,
        default=True,
        help="generate self-signed certificate",
    )

    parser.add_argument(
        "--content_vec_500", type=str, help="path to content_vec_500 model(pytorch)"
    )
    parser.add_argument(
        "--content_vec_500_onnx", type=str, help="path to content_vec_500 model(onnx)"
    )
    parser.add_argument(
        "--content_vec_500_onnx_on",
        type=strtobool,
        default=False,
        help="use or not onnx for  content_vec_500",
    )
    parser.add_argument(
        "--hubert_base", type=str, help="path to hubert_base model(pytorch)"
    )
    parser.add_argument(
        "--hubert_base_jp", type=str, help="path to hubert_base_jp model(pytorch)"
    )
    parser.add_argument(
        "--hubert_soft", type=str, help="path to hubert_soft model(pytorch)"
    )
    parser.add_argument(
        "--nsf_hifigan", type=str, help="path to nsf_hifigan model(pytorch)"
    )

    return parser


def printMessage(message, level=0):
    pf = platform.system()
    if pf == "Windows":
        if level == 0:
            print(f"{message}")
        elif level == 1:
            print(f"    {message}")
        elif level == 2:
            print(f"    {message}")
        else:
            print(f"    {message}")
    else:
        if level == 0:
            print(f"\033[17m{message}\033[0m")
        elif level == 1:
            print(f"\033[34m    {message}\033[0m")
        elif level == 2:
            print(f"\033[32m    {message}\033[0m")
        else:
            print(f"\033[47m    {message}\033[0m")


def downloadWeight():
    voiceChangerParams = VoiceChangerParams(
        content_vec_500=args.content_vec_500,
        content_vec_500_onnx=args.content_vec_500_onnx,
        content_vec_500_onnx_on=args.content_vec_500_onnx_on,
        hubert_base=args.hubert_base,
        hubert_base_jp=args.hubert_base_jp,
        hubert_soft=args.hubert_soft,
        nsf_hifigan=args.nsf_hifigan,
    )

    # file exists check (currently only for rvc)
    downloadParams = []
    if os.path.exists(voiceChangerParams.hubert_base) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/ddPn08/rvc-webui-models/resolve/main/embeddings/hubert_base.pt",
                "saveTo": voiceChangerParams.hubert_base,
                "position": 0,
            }
        )
    if os.path.exists(voiceChangerParams.hubert_base_jp) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/rinna/japanese-hubert-base/resolve/main/fairseq/model.pt",
                "saveTo": voiceChangerParams.hubert_base_jp,
                "position": 1,
            }
        )
    if os.path.exists(voiceChangerParams.hubert_soft) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/resolve/main/ddsp-svc30/embedder/hubert-soft-0d54a1f4.pt",
                "saveTo": voiceChangerParams.hubert_soft,
                "position": 2,
            }
        )
    if os.path.exists(voiceChangerParams.nsf_hifigan) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/resolve/main/ddsp-svc30/nsf_hifigan_20221211/model.bin",
                "saveTo": voiceChangerParams.nsf_hifigan,
                "position": 3,
            }
        )
    nsf_hifigan_config = os.path.join(
        os.path.dirname(voiceChangerParams.nsf_hifigan), "config.json"
    )

    if os.path.exists(nsf_hifigan_config) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/raw/main/ddsp-svc30/nsf_hifigan_20221211/config.json",
                "saveTo": nsf_hifigan_config,
                "position": 4,
            }
        )

    with ThreadPoolExecutor() as pool:
        pool.map(download, downloadParams)

    if (
        os.path.exists(voiceChangerParams.hubert_base) is False
        or os.path.exists(voiceChangerParams.hubert_base_jp) is False
    ):
        printMessage("RVC用のモデルファイルのダウンロードに失敗しました。", level=2)
        printMessage("failed to download weight for rvc", level=2)


parser = setupArgParser()
args, unknown = parser.parse_known_args()

printMessage(f"Booting PHASE :{__name__}", level=2)

PORT = args.p


def localServer():
    uvicorn.run(
        f"{os.path.basename(__file__)[:-3]}:app_socketio",
        host="0.0.0.0",
        port=int(PORT),
        reload=False if hasattr(sys, "_MEIPASS") else True,
        log_level="warning",
    )


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
        print(e)


if __name__ == "MMVCServerSIO":
    voiceChangerParams = VoiceChangerParams(
        content_vec_500=args.content_vec_500,
        content_vec_500_onnx=args.content_vec_500_onnx,
        content_vec_500_onnx_on=args.content_vec_500_onnx_on,
        hubert_base=args.hubert_base,
        hubert_base_jp=args.hubert_base_jp,
        hubert_soft=args.hubert_soft,
        nsf_hifigan=args.nsf_hifigan,
    )

    # file exists check (currently only for rvc)
    downloadParams = []
    if os.path.exists(voiceChangerParams.hubert_base) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/ddPn08/rvc-webui-models/resolve/main/embeddings/hubert_base.pt",
                "saveTo": voiceChangerParams.hubert_base,
                "position": 0,
            }
        )
    if os.path.exists(voiceChangerParams.hubert_base_jp) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/rinna/japanese-hubert-base/resolve/main/fairseq/model.pt",
                "saveTo": voiceChangerParams.hubert_base_jp,
                "position": 1,
            }
        )
    if os.path.exists(voiceChangerParams.hubert_soft) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/resolve/main/ddsp-svc30/embedder/hubert-soft-0d54a1f4.pt",
                "saveTo": voiceChangerParams.hubert_soft,
                "position": 2,
            }
        )
    if os.path.exists(voiceChangerParams.nsf_hifigan) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/resolve/main/ddsp-svc30/nsf_hifigan_20221211/model.bin",
                "saveTo": voiceChangerParams.nsf_hifigan,
                "position": 3,
            }
        )
    nsf_hifigan_config = os.path.join(
        os.path.dirname(voiceChangerParams.nsf_hifigan), "config.json"
    )

    if os.path.exists(nsf_hifigan_config) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/raw/main/ddsp-svc30/nsf_hifigan_20221211/config.json",
                "saveTo": nsf_hifigan_config,
                "position": 4,
            }
        )

    with ThreadPoolExecutor() as pool:
        pool.map(download, downloadParams)

    if (
        os.path.exists(voiceChangerParams.hubert_base) is False
        or os.path.exists(voiceChangerParams.hubert_base_jp) is False
    ):
        printMessage("RVC用のモデルファイルのダウンロードに失敗しました。", level=2)
        printMessage("failed to download weight for rvc", level=2)

    voiceChangerManager = VoiceChangerManager.get_instance(voiceChangerParams)
    app_fastapi = MMVC_Rest.get_instance(voiceChangerManager)
    app_socketio = MMVC_SocketIOApp.get_instance(app_fastapi, voiceChangerManager)


if __name__ == "__mp_main__":
    printMessage("サーバプロセスを起動しています。", level=2)

if __name__ == "__main__":
    mp.freeze_support()

    printMessage("Voice Changerを起動しています。", level=2)

    downloadWeight()

    PORT = args.p

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
        printMessage(
            f"protocol: HTTPS(self-signed), key:{key_path}, cert:{cert_path}", level=1
        )

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
    printMessage("ブラウザで次のURLを開いてください.", level=2)
    if args.https == 1:
        printMessage("https://<IP>:<PORT>/", level=1)
    else:
        printMessage("http://<IP>:<PORT>/", level=1)

    printMessage("多くの場合は次のいずれかのURLにアクセスすると起動します。", level=2)
    if "EX_PORT" in locals() and "EX_IP" in locals():  # シェルスクリプト経由起動(docker)
        if args.https == 1:
            printMessage(f"https://localhost:{EX_PORT}/", level=1)
            for ip in EX_IP.strip().split(" "):
                printMessage(f"https://{ip}:{EX_PORT}/", level=1)
        else:
            printMessage(f"http://localhost:{EX_PORT}/", level=1)
    else:  # 直接python起動
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        hostname = s.getsockname()[0]
        if args.https == 1:
            printMessage(f"https://localhost:{PORT}/", level=1)
            printMessage(f"https://{hostname}:{PORT}/", level=1)
        else:
            printMessage(f"http://localhost:{PORT}/", level=1)

    # サーバ起動
    if args.https:
        # HTTPS サーバ起動
        uvicorn.run(
            f"{os.path.basename(__file__)[:-3]}:app_socketio",
            host="0.0.0.0",
            port=int(PORT),
            reload=False if hasattr(sys, "_MEIPASS") else True,
            ssl_keyfile=key_path,
            ssl_certfile=cert_path,
            # log_level="warning"
        )
    else:
        p = mp.Process(name="p", target=localServer)
        p.start()
        try:
            if sys.platform.startswith("win"):
                process = subprocess.Popen(
                    [NATIVE_CLIENT_FILE_WIN, "-u", f"http://localhost:{PORT}/"]
                )
                return_code = process.wait()
                print("client closed.")
                p.terminate()
            elif sys.platform.startswith("darwin"):
                process = subprocess.Popen(
                    [NATIVE_CLIENT_FILE_MAC, "-u", f"http://localhost:{PORT}/"]
                )
                return_code = process.wait()
                print("client closed.")
                p.terminate()

        except Exception as e:
            print(e)
