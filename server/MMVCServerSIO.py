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
import webbrowser
from mods.ssl import create_self_signed_cert
from voice_changer.VoiceChangerManager import VoiceChangerManager
from sio.MMVC_SocketIOApp import MMVC_SocketIOApp
from restapi.MMVC_Rest import MMVC_Rest
from const import NATIVE_CLIENT_FILE_MAC, NATIVE_CLIENT_FILE_WIN, SSL_KEY_DIR, setModelType
import subprocess
import multiprocessing as mp


def setupArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str, default="MMVC",
                        help="Server type. MMVC|TRAIN")
    parser.add_argument("-p", type=int, default=18888, help="port")
    parser.add_argument("-c", type=str, help="path for the config.json")
    parser.add_argument("-m", type=str, help="path for the model file")
    parser.add_argument("-o", type=str, help="path for the onnx model file")
    parser.add_argument("--https", type=strtobool,
                        default=False, help="use https")
    parser.add_argument("--httpsKey", type=str,
                        default="ssl.key", help="path for the key of https")
    parser.add_argument("--httpsCert", type=str,
                        default="ssl.cert", help="path for the cert of https")
    parser.add_argument("--httpsSelfSigned", type=strtobool,
                        default=True, help="generate self-signed certificate")
    parser.add_argument("--colab", type=strtobool,
                        default=False, help="run on colab")
    parser.add_argument("--modelType", type=str,
                        default="MMVCv15", help="model type: MMVCv13, MMVCv15, so-vits-svc-40v2, so-vits-svc-40v2_tsukuyomi")
    parser.add_argument("--cluster", type=str, help="path to cluster model")
    parser.add_argument("--hubert", type=str, help="path to hubert model")
    parser.add_argument("--internal", type=strtobool, default=False, help="各種パスをmac appの中身に変換")

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

# printMessage(f"Phase name:{__name__}", level=2)
# thisFilename = os.path.basename(__file__)[:-3]

# if __name__ == thisFilename or args.colab == True:
# printMessage(f"PHASE3:{__name__}", level=2)

printMessage(f"Booting PHASE :{__name__}", level=2)

TYPE = args.t
PORT = args.p
CONFIG = args.c if args.c != None else None
MODEL = args.m if args.m != None else None
ONNX_MODEL = args.o if args.o != None else None
HUBERT_MODEL = args.hubert if args.hubert != None else None  # hubertはユーザがダウンロードして解凍フォルダに格納する運用。
CLUSTER_MODEL = args.cluster if args.cluster != None else None
if args.internal and hasattr(sys, "_MEIPASS"):
    print("use internal path")
    if CONFIG != None:
        CONFIG = os.path.join(sys._MEIPASS, CONFIG)
    if MODEL != None:
        MODEL = os.path.join(sys._MEIPASS, MODEL)
    if ONNX_MODEL:
        ONNX_MODEL = os.path.join(sys._MEIPASS, ONNX_MODEL)
    if CLUSTER_MODEL:
        CLUSTER_MODEL = os.path.join(sys._MEIPASS, CLUSTER_MODEL)
    print(" config path:", CONFIG)
    print(" model path:", MODEL)
    print(" onnx model path:", ONNX_MODEL)
    print(" cluster model path:", CLUSTER_MODEL)


MODEL_TYPE = os.environ.get('MODEL_TYPE', None)
if MODEL_TYPE == None:
    MODEL_TYPE = args.modelType
setModelType(MODEL_TYPE)


def localServer():
    uvicorn.run(
        f"{os.path.basename(__file__)[:-3]}:app_socketio",
        host="0.0.0.0",
        port=int(PORT),
        reload=False if hasattr(sys, "_MEIPASS") else True,
        log_level="warning"
    )


if args.colab == True:
    os.environ["colab"] = "True"

if __name__ == 'MMVCServerSIO':
    voiceChangerManager = VoiceChangerManager.get_instance({"hubert": HUBERT_MODEL})
    if CONFIG and (MODEL or ONNX_MODEL):
        if MODEL_TYPE == "MMVCv15" or MODEL_TYPE == "MMVCv13":
            voiceChangerManager.loadModel(CONFIG, MODEL, ONNX_MODEL, None)
        else:
            voiceChangerManager.loadModel(CONFIG, MODEL, ONNX_MODEL, CLUSTER_MODEL)

    app_fastapi = MMVC_Rest.get_instance(voiceChangerManager)
    app_socketio = MMVC_SocketIOApp.get_instance(app_fastapi, voiceChangerManager)


if __name__ == '__mp_main__':
    printMessage(f"サーバプロセスを起動しています。", level=2)

if __name__ == '__main__':
    mp.freeze_support()

    printMessage(f"Voice Changerを起動しています。", level=2)
    TYPE = args.t
    PORT = args.p
    CONFIG = args.c
    MODEL = args.m if args.m != None else None
    ONNX_MODEL = args.o if args.o != None else None
    if TYPE != "MMVC" and TYPE != "TRAIN":
        print("Type(-t) should be MMVC or TRAIN")
        exit(1)

    # printMessage(f"Start MMVC SocketIO Server", level=0)
    printMessage(f"-- 設定 -- ", level=1)
    printMessage(f"CONFIG:{CONFIG}, MODEL:{MODEL} ONNX_MODEL:{ONNX_MODEL}", level=1)

    if args.colab == False:
        if os.getenv("EX_PORT"):
            EX_PORT = os.environ["EX_PORT"]
            printMessage(
                f"External_Port:{EX_PORT} Internal_Port:{PORT}", level=1)
        else:
            printMessage(f"Internal_Port:{PORT}", level=1)

        if os.getenv("EX_TB_PORT"):
            EX_TB_PORT = os.environ["EX_TB_PORT"]
            printMessage(f"External_TeonsorBord_Port:{EX_TB_PORT}", level=1)

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

        if TYPE == "MMVC":
            path = ""
        else:
            path = "trainer"

        printMessage(f"多くの場合は次のいずれかのURLにアクセスすると起動します。", level=2)
        if "EX_PORT" in locals() and "EX_IP" in locals():  # シェルスクリプト経由起動(docker)
            if args.https == 1:
                printMessage(f"https://localhost:{EX_PORT}/{path}", level=1)
                for ip in EX_IP.strip().split(" "):
                    printMessage(f"https://{ip}:{EX_PORT}/{path}", level=1)
            else:
                printMessage(f"http://localhost:{EX_PORT}/{path}", level=1)
        else:  # 直接python起動
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            hostname = s.getsockname()[0]
            if args.https == 1:
                printMessage(f"https://localhost:{PORT}/{path}", level=1)
                printMessage(f"https://{hostname}:{PORT}/{path}", level=1)
            else:
                printMessage(f"http://localhost:{PORT}/{path}", level=1)

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
        # HTTP サーバ起動
        if args.colab == True:
            uvicorn.run(
                f"{os.path.basename(__file__)[:-3]}:app_fastapi",
                host="0.0.0.0",
                port=int(PORT),
                log_level="warning"
            )
        else:
            # uvicorn.run(
            #     f"{os.path.basename(__file__)[:-3]}:app_socketio",
            #     host="0.0.0.0",
            #     port=int(PORT),
            #     reload = False if hasattr(sys, "_MEIPASS") else True,
            #     log_level="warning"
            # )
            os.environ['MODEL_TYPE'] = MODEL_TYPE
            p = mp.Process(name="p", target=localServer)
            p.start()
            try:
                if sys.platform.startswith('win'):
                    # process = subprocess.Popen(["voice-changer-native-client.exe", "-u", f"http://localhost:{PORT}/{path}"])
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
