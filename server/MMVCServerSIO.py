import sys, os, struct, argparse, shutil, traceback, base64, struct
import numpy as np
import misc.log_control

from dataclasses import dataclass
from datetime import datetime
from distutils.util import strtobool

from scipy.io.wavfile import write, read

sys.path.append("MMVC_Trainer")
sys.path.append("MMVC_Trainer/text")

from fastapi.routing import APIRoute
from fastapi import HTTPException,  FastAPI, UploadFile, File, Form

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

import uvicorn
import socketio

from typing import Callable
from mods.ssl import create_self_signed_cert
from voice_changer.VoiceChangerManager import VoiceChangerManager
from sio.MMVC_SocketIOApp import MMVC_SocketIOApp

from restapi.MMVC_Rest import MMVC_Rest


def setupArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str, default="MMVC",
                        help="Server type. MMVC|TRAIN")
    parser.add_argument("-p", type=int, default=8080, help="port")
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
    return parser


def printMessage(message, level=0):
    if level == 0:
        print(f"\033[17m{message}\033[0m")
    elif level == 1:
        print(f"\033[34m    {message}\033[0m")
    elif level == 2:
        print(f"\033[32m    {message}\033[0m")
    else:
        print(f"\033[47m    {message}\033[0m")



global app_socketio
global app_fastapi

parser = setupArgParser()
args = parser.parse_args()

printMessage(f"Phase name:{__name__}", level=2)
thisFilename = os.path.basename(__file__)[:-3]

if __name__ == thisFilename or args.colab == True:
    printMessage(f"PHASE3:{__name__}", level=2)
    TYPE = args.t
    PORT = args.p
    CONFIG = args.c
    MODEL = args.m if args.m != None else None
    ONNX_MODEL = args.o if args.o != None else None


    if args.colab == True:
        os.environ["colab"] = "True"
    # if os.getenv("EX_TB_PORT"):
    #     EX_TB_PORT = os.environ["EX_TB_PORT"]
    #     exApplitionInfo.external_tensorboard_port = int(EX_TB_PORT)

    voiceChangerManager = VoiceChangerManager.get_instance()    
    if CONFIG and (MODEL or ONNX_MODEL):
        voiceChangerManager.loadModel(CONFIG, MODEL, ONNX_MODEL)
    app_fastapi = MMVC_Rest.get_instance(voiceChangerManager)
    app_socketio = MMVC_SocketIOApp.get_instance(app_fastapi, voiceChangerManager)


if __name__ == '__mp_main__':
    printMessage(f"PHASE2:{__name__}", level=2)

if __name__ == '__main__':
    printMessage(f"PHASE1:{__name__}", level=2)
    TYPE = args.t
    PORT = args.p
    CONFIG = args.c
    MODEL = args.m if args.m != None else None
    ONNX_MODEL = args.o if args.o != None else None
    if TYPE != "MMVC" and TYPE != "TRAIN":
        print("Type(-t) should be MMVC or TRAIN")
        exit(1)

    printMessage(f"Start MMVC SocketIO Server", level=0)
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

        # HTTPS key/cert??????
        if args.https and args.httpsSelfSigned == 1:
            # HTTPS(???????????????????????????)
            os.makedirs("./key", exist_ok=True)
            key_base_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            keyname = f"{key_base_name}.key"
            certname = f"{key_base_name}.cert"
            create_self_signed_cert(certname, keyname, certargs={"Country": "JP",
                                                                 "State": "Tokyo",
                                                                 "City": "Chuo-ku",
                                                                 "Organization": "F",
                                                                 "Org. Unit": "F"}, cert_dir="./key")
            key_path = os.path.join("./key", keyname)
            cert_path = os.path.join("./key", certname)
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

        # ??????????????????
        if args.https == 1:
            printMessage(
                f"open https://<IP>:<PORT>/ with your browser.", level=0)
        else:
            printMessage(
                f"open http://<IP>:<PORT>/ with your browser.", level=0)

        if TYPE == "MMVC":
            path = ""
        else:
            path = "trainer"
        if "EX_PORT" in locals() and "EX_IP" in locals() and args.https == 1:
            printMessage(f"In many cases it is one of the following", level=1)
            printMessage(f"https://localhost:{EX_PORT}/{path}", level=1)
            for ip in EX_IP.strip().split(" "):
                printMessage(f"https://{ip}:{EX_PORT}/{path}", level=1)
        elif "EX_PORT" in locals() and "EX_IP" in locals() and args.https == 0:
            printMessage(f"In many cases it is one of the following", level=1)
            printMessage(f"http://localhost:{EX_PORT}/{path}", level=1)

    # ???????????????
    if args.https:
        # HTTPS ???????????????
        res = uvicorn.run(
            f"{os.path.basename(__file__)[:-3]}:app_socketio",
            host="0.0.0.0",
            port=int(PORT),
            reload=True,
            ssl_keyfile=key_path,
            ssl_certfile=cert_path,
            # log_level="warning"
        )
    else:
        # HTTP ???????????????
        if args.colab == True:
            uvicorn.run(
                f"{os.path.basename(__file__)[:-3]}:app_fastapi",
                host="0.0.0.0",
                port=int(PORT),
                log_level="warning"
            )
        else:
            uvicorn.run(
                f"{os.path.basename(__file__)[:-3]}:app_socketio",
                host="0.0.0.0",
                port=int(PORT),
                reload=True,
                log_level="warning"
            )

