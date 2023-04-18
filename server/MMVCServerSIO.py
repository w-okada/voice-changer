import sys

from distutils.util import strtobool
from datetime import datetime
from dataclasses import dataclass
import misc.log_control
import socket
import platform
import os
import argparse
import uvicorn
from mods.ssl import create_self_signed_cert
from voice_changer.VoiceChangerManager import VoiceChangerManager
from sio.MMVC_SocketIOApp import MMVC_SocketIOApp
from restapi.MMVC_Rest import MMVC_Rest
from const import NATIVE_CLIENT_FILE_MAC, NATIVE_CLIENT_FILE_WIN, SSL_KEY_DIR
import subprocess
import multiprocessing as mp


def setupArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=int, default=18888, help="port")
    parser.add_argument("--https", type=strtobool,
                        default=False, help="use https")
    parser.add_argument("--httpsKey", type=str,
                        default="ssl.key", help="path for the key of https")
    parser.add_argument("--httpsCert", type=str,
                        default="ssl.cert", help="path for the cert of https")
    parser.add_argument("--httpsSelfSigned", type=strtobool,
                        default=True, help="generate self-signed certificate")

    # parser.add_argument("--internal", type=strtobool, default=False, help="各種パスをmac appの中身に変換")

    parser.add_argument("--content_vec_500", type=str, help="path to content_vec_500 model(pytorch)")
    parser.add_argument("--content_vec_500_onnx", type=str, help="path to content_vec_500 model(onnx)")
    parser.add_argument("--content_vec_500_onnx_on", type=strtobool, default=False, help="use or not onnx for  content_vec_500")
    parser.add_argument("--hubert_base", type=str, help="path to hubert_base model(pytorch)")
    parser.add_argument("--hubert_soft", type=str, help="path to hubert_soft model(pytorch)")
    parser.add_argument("--nsf_hifigan", type=str, help="path to nsf_hifigan model(pytorch)")

    return parser


def printMessage(message, level=0):
    pf = platform.system()
    if pf == 'Windows':
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
        log_level="warning"
    )


if __name__ == 'MMVCServerSIO':
    voiceChangerManager = VoiceChangerManager.get_instance({
        "content_vec_500": args.content_vec_500,
        "content_vec_500_onnx": args.content_vec_500_onnx,
        "content_vec_500_onnx_on": args.content_vec_500_onnx_on,
        "hubert_base": args.hubert_base,
        "hubert_soft": args.hubert_soft,
        "nsf_hifigan": args.nsf_hifigan,
    })

    app_fastapi = MMVC_Rest.get_instance(voiceChangerManager)
    app_socketio = MMVC_SocketIOApp.get_instance(app_fastapi, voiceChangerManager)


if __name__ == '__mp_main__':
    printMessage(f"サーバプロセスを起動しています。", level=2)

if __name__ == '__main__':
    mp.freeze_support()

    printMessage(f"Voice Changerを起動しています。", level=2)
    PORT = args.p

    if os.getenv("EX_PORT"):
        EX_PORT = os.environ["EX_PORT"]
        printMessage(
            f"External_Port:{EX_PORT} Internal_Port:{PORT}", level=1)
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
        create_self_signed_cert(certname, keyname, certargs={"Country": "JP",
                                                             "State": "Tokyo",
                                                             "City": "Chuo-ku",
                                                             "Organization": "F",
                                                             "Org. Unit": "F"}, cert_dir=SSL_KEY_DIR)
        key_path = os.path.join(SSL_KEY_DIR, keyname)
        cert_path = os.path.join(SSL_KEY_DIR, certname)
        printMessage(
            f"protocol: HTTPS(self-signed), key:{key_path}, cert:{cert_path}", level=1)

    elif args.https and args.httpsSelfSigned == 0:
        # HTTPS
        key_path = args.httpsKey
        cert_path = args.httpsCert
        printMessage(
            f"protocol: HTTPS, key:{key_path}, cert:{cert_path}", level=1)
    else:
        # HTTP
        printMessage(f"protocol: HTTP", level=1)
    printMessage(f"-- ---- -- ", level=1)

    # アドレス表示
    printMessage(
        f"ブラウザで次のURLを開いてください.", level=2)
    if args.https == 1:
        printMessage(
            f"https://<IP>:<PORT>/", level=1)
    else:
        printMessage(
            f"http://<IP>:<PORT>/", level=1)

    printMessage(f"多くの場合は次のいずれかのURLにアクセスすると起動します。", level=2)
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
        res = uvicorn.run(
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
            if sys.platform.startswith('win'):
                process = subprocess.Popen([NATIVE_CLIENT_FILE_WIN, "-u", f"http://localhost:{PORT}/{path}"])
                return_code = process.wait()
                print("client closed.")
                p.terminate()
            elif sys.platform.startswith('darwin'):
                process = subprocess.Popen([NATIVE_CLIENT_FILE_MAC, "-u", f"http://localhost:{PORT}/{path}"])
                return_code = process.wait()
                print("client closed.")
                p.terminate()

        except Exception as e:
            print(e)
