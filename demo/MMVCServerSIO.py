import sys, os, struct, argparse, logging, shutil,  base64, traceback
sys.path.append("/MMVC_Trainer")
sys.path.append("/MMVC_Trainer/text")

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from scipy.io.wavfile import write, read

import socketio
from distutils.util import strtobool
from datetime import datetime

import torch
import numpy as np

from mods.ssl import create_self_signed_cert
from mods.VoiceChanger import VoiceChanger

class UvicornSuppressFilter(logging.Filter):
    def filter(self, record):
        return False

logger = logging.getLogger("uvicorn.error")
logger.addFilter(UvicornSuppressFilter())
# logger.propagate = False
logger = logging.getLogger("multipart.multipart")
logger.propagate = False



class VoiceModel(BaseModel):
    gpu: int
    srcId: int
    dstId: int
    timestamp: int
    buffer: str


class MyCustomNamespace(socketio.AsyncNamespace): 
    def __init__(self, namespace):
        super().__init__(namespace)

    def loadModel(self, config, model):
        if hasattr(self, 'voiceChanger') == True:
            self.voiceChanger.destroy()
        self.voiceChanger = VoiceChanger(config, model)

    def changeVoice(self, gpu, srcId, dstId, timestamp, unpackedData):
        return self.voiceChanger.on_request(gpu, srcId, dstId, timestamp, unpackedData)

    def on_connect(self, sid, environ):
        # print('[{}] connet sid : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S') , sid))
        pass

    async def on_request_message(self, sid, msg): 
        # print("on_request_message", torch.cuda.memory_allocated())
        gpu = int(msg[0])
        srcId = int(msg[1])
        dstId = int(msg[2])
        timestamp = int(msg[3])
        data = msg[4]
        # print(srcId, dstId, timestamp)
        unpackedData = np.array(struct.unpack('<%sh'%(len(data) // struct.calcsize('<h') ), data))
        audio1 = self.changeVoice(gpu, srcId, dstId, timestamp, unpackedData)

        bin = struct.pack('<%sh'%len(audio1), *audio1)

        await self.emit('response',[timestamp, bin])

    def on_disconnect(self, sid):
        # print('[{}] disconnect'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        pass;


def setupArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=int, default=8080, help="port")
    parser.add_argument("-c", type=str, help="path for the config.json")
    parser.add_argument("-m", type=str, help="path for the model file")
    parser.add_argument("--https", type=strtobool, default=False, help="use https")
    parser.add_argument("--httpsKey", type=str, default="ssl.key", help="path for the key of https")
    parser.add_argument("--httpsCert", type=str, default="ssl.cert", help="path for the cert of https")
    parser.add_argument("--httpsSelfSigned", type=strtobool, default=True, help="generate self-signed certificate")
    parser.add_argument("--colab", type=strtobool, default=False, help="run on colab")
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


printMessage(f"Phase name:{__name__}", level=2)
thisFilename = os.path.basename(__file__)[:-3]


if __name__ == thisFilename:
    printMessage(f"PHASE3:{__name__}", level=2)
    parser = setupArgParser()
    args = parser.parse_args()
    PORT = args.p
    CONFIG = args.c
    MODEL  = args.m

    app_fastapi = FastAPI()
    sio = socketio.AsyncServer(
        async_mode='asgi',
        cors_allowed_origins='*'
    )
    namespace = MyCustomNamespace('/test')
    sio.register_namespace(namespace) 
    if CONFIG and MODEL:
        namespace.loadModel(CONFIG, MODEL)
    app_socketio = socketio.ASGIApp(
        sio, 
        other_asgi_app=app_fastapi,
        static_files={
            '': '../frontend/dist',
            '/': '../frontend/dist/index.html',
        }
    )

    @app_fastapi.get("/api/hello")
    async def index():
        return {"result": "Index"}
    

    @app_fastapi.post("/upload_model_file")
    async def upload_file(configFile:UploadFile = File(...), modelFile: UploadFile = File(...)):
        if configFile and modelFile:
            for file in [modelFile, configFile]:
                filename = file.filename
                fileobj = file.file
                upload_dir = open(os.path.join(".", filename),'wb+')
                shutil.copyfileobj(fileobj, upload_dir)
                upload_dir.close()
            namespace.loadModel(configFile.filename, modelFile.filename)                
            return {"uploaded files": f"{configFile.filename}, {modelFile.filename} "}
        return {"Error": "uploaded file is not found."}



    @app_fastapi.post("/test")
    async def post_test(voice:VoiceModel):
        try:
            # print("POST REQUEST PROCESSING....")
            gpu = voice.gpu
            srcId = voice.srcId
            dstId = voice.dstId
            timestamp = voice.timestamp
            buffer = voice.buffer
            wav = base64.b64decode(buffer)

            if wav==0:
                samplerate, data=read("dummy.wav")
                unpackedData = data
            else:
                unpackedData = np.array(struct.unpack('<%sh'%(len(wav) // struct.calcsize('<h') ), wav))
                write("logs/received_data.wav", 24000, unpackedData.astype(np.int16))

            changedVoice = namespace.changeVoice(gpu, srcId, dstId, timestamp, unpackedData)
            changedVoiceBase64 = base64.b64encode(changedVoice).decode('utf-8')

            data = {
                "gpu":gpu,
                "srcId":srcId,
                "dstId":dstId,
                "timestamp":timestamp,
                "changedVoiceBase64":changedVoiceBase64
            }

            json_compatible_item_data = jsonable_encoder(data)
            
            return JSONResponse(content=json_compatible_item_data)
        except Exception as e:
            print("REQUEST PROCESSING!!!! EXCEPTION!!!", e)
            print(traceback.format_exc())
            return str(e)


if __name__ == '__mp_main__':
    printMessage(f"PHASE2:{__name__}", level=2)


if __name__ == '__main__':
    printMessage(f"PHASE1:{__name__}", level=2)

    parser = setupArgParser()
    args = parser.parse_args()
    PORT = args.p
    CONFIG = args.c
    MODEL  = args.m

    printMessage(f"Start MMVC SocketIO Server", level=0)
    printMessage(f"CONFIG:{CONFIG}, MODEL:{MODEL}", level=1)

    if args.colab == False:
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
          os.makedirs("./key", exist_ok=True)
          key_base_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
          keyname = f"{key_base_name}.key"
          certname = f"{key_base_name}.cert"
          create_self_signed_cert(certname, keyname, certargs=
                              {"Country": "JP",
                                  "State": "Tokyo",
                                  "City": "Chuo-ku",
                                  "Organization": "F",
                                  "Org. Unit": "F"}, cert_dir="./key")
          key_path = os.path.join("./key", keyname)
          cert_path = os.path.join("./key", certname)
          printMessage(f"protocol: HTTPS(self-signed), key:{key_path}, cert:{cert_path}", level=1)
      elif args.https and args.httpsSelfSigned == 0:
          # HTTPS 
          key_path = args.httpsKey
          cert_path = args.httpsCert
          printMessage(f"protocol: HTTPS, key:{key_path}, cert:{cert_path}", level=1)
      else:
          # HTTP
          printMessage(f"protocol: HTTP", level=1)

      # アドレス表示
      if args.https == 1:
          printMessage(f"open https://<IP>:<PORT>/ with your browser.", level=0)
      else:
          printMessage(f"open http://<IP>:<PORT>/ with your browser.", level=0)
      
      if EX_PORT and EX_IP and args.https == 1:
          printMessage(f"In many cases it is one of the following", level=1)
          printMessage(f"https://localhost:{EX_PORT}/", level=1)
          for ip in EX_IP.strip().split(" "):
              printMessage(f"https://{ip}:{EX_PORT}/", level=1)
      elif EX_PORT and EX_IP and args.https == 0:
          printMessage(f"In many cases it is one of the following", level=1)
          printMessage(f"http://localhost:{EX_PORT}/", level=1)


    # サーバ起動
    if args.https:
        # HTTPS サーバ起動 
        uvicorn.run(
            f"{os.path.basename(__file__)[:-3]}:app_socketio", 
            host="0.0.0.0", 
            port=int(PORT), 
            reload=True, 
            ssl_keyfile = key_path,
            ssl_certfile = cert_path,
            log_level="critical"            
        )
    else:
        # HTTP サーバ起動
        uvicorn.run(
            f"{os.path.basename(__file__)[:-3]}:app_socketio", 
            host="0.0.0.0", 
            port=int(PORT), 
            reload=True,
            log_level="critical"
        )
