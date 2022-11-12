import sys, os, struct, argparse, logging, shutil,  base64, traceback
from dataclasses import dataclass
sys.path.append("/MMVC_Trainer")
sys.path.append("/MMVC_Trainer/text")

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scipy.io.wavfile import write, read

import socketio
from distutils.util import strtobool
from datetime import datetime

import torch
import numpy as np


from mods.ssl import create_self_signed_cert
from mods.VoiceChanger import VoiceChanger


# File Uploader
from mods.FileUploader import upload_file, concat_file_chunks

# Trainer Rest Internal 
from mods.Trainer_Speakers import mod_get_speakers 
from mods.Trainer_Speaker import mod_delete_speaker
from mods.Trainer_Speaker_Voices import mod_get_speaker_voices
from mods.Trainer_Speaker_Voice import mod_get_speaker_voice
from mods.Trainer_MultiSpeakerSetting import mod_get_multi_speaker_setting, mod_post_multi_speaker_setting
from mods.Trainer_Models import mod_get_models
from mods.Trainer_Model import mod_get_model, mod_delete_model
from mods.Trainer_Training import mod_post_pre_training, mod_post_start_training, mod_post_stop_training, mod_get_related_files, mod_get_tail_training_log

class UvicornSuppressFilter(logging.Filter):
    def filter(self, record):
        return False

logger = logging.getLogger("uvicorn.error")
logger.addFilter(UvicornSuppressFilter())
# logger.propagate = False
logger = logging.getLogger("multipart.multipart")
logger.propagate = False

@dataclass
class ExApplicationInfo():
    external_tensorboard_port:int


exApplitionInfo = ExApplicationInfo(external_tensorboard_port=0)

class VoiceModel(BaseModel):
    gpu: int
    srcId: int
    dstId: int
    timestamp: int
    prefixChunkSize: int
    buffer: str


class MyCustomNamespace(socketio.AsyncNamespace): 
    def __init__(self, namespace):
        super().__init__(namespace)

    def loadModel(self, config, model):
        if hasattr(self, 'voiceChanger') == True:
            self.voiceChanger.destroy()
        self.voiceChanger = VoiceChanger(config, model)

    # def loadWhisperModel(self, model):
    #     self.whisper = Whisper()
    #     self.whisper.loadModel("tiny")
    #     print("load")

    def changeVoice(self, gpu, srcId, dstId, timestamp, prefixChunkSize, unpackedData):
        # if hasattr(self, 'whisper') == True:
        #     self.whisper.addData(unpackedData)
        if hasattr(self, 'voiceChanger') == True:
            return self.voiceChanger.on_request(gpu, srcId, dstId, timestamp, prefixChunkSize, unpackedData)
        else:
            print("Voice Change is not loaded. Did you load a correct model?")
            return np.zeros(1).astype(np.int16)


    # def transcribe(self):
    #     if hasattr(self, 'whisper') == True:
    #         self.whisper.transcribe(0)
    #     else:
    #         print("whisper not found")


    def on_connect(self, sid, environ):
        # print('[{}] connet sid : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S') , sid))
        pass

    async def on_request_message(self, sid, msg): 
        # print("on_request_message", torch.cuda.memory_allocated())
        gpu = int(msg[0])
        srcId = int(msg[1])
        dstId = int(msg[2])
        timestamp = int(msg[3])
        prefixChunkSize = int(msg[4])
        data = msg[5]
        # print(srcId, dstId, timestamp)
        unpackedData = np.array(struct.unpack('<%sh'%(len(data) // struct.calcsize('<h') ), data))
        audio1 = self.changeVoice(gpu, srcId, dstId, timestamp, prefixChunkSize, unpackedData)

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
global app_fastapi

parser = setupArgParser()
args = parser.parse_args()

printMessage(f"Phase name:{__name__}", level=2)
thisFilename = os.path.basename(__file__)[:-3]

from typing import Callable, List
from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
class ValidationErrorLoggingRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            try:
                return await original_route_handler(request)
            except Exception as exc:
                print("Exception", request.url, str(exc))
                body = await request.body()
                detail = {"errors": exc.errors(), "body": body.decode()}
                raise HTTPException(status_code=422, detail=detail)

        return custom_route_handler

if __name__ == thisFilename or args.colab == True:
    printMessage(f"PHASE3:{__name__}", level=2)
    PORT = args.p
    CONFIG = args.c
    MODEL  = args.m

    app_fastapi = FastAPI()
    app_fastapi.router.route_class = ValidationErrorLoggingRoute
    app_fastapi.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app_fastapi.mount("/front", StaticFiles(directory="../frontend/dist", html=True), name="static")

    app_fastapi.mount("/trainer", StaticFiles(directory="../frontend/dist", html=True), name="static")

    app_fastapi.mount("/recorder", StaticFiles(directory="../frontend/dist", html=True), name="static")

    sio = socketio.AsyncServer(
        async_mode='asgi',
        cors_allowed_origins='*'
    )
    namespace = MyCustomNamespace('/test')
    sio.register_namespace(namespace) 
    if CONFIG and MODEL:
        namespace.loadModel(CONFIG, MODEL)
    # namespace.loadWhisperModel("base")
    
    
    app_socketio = socketio.ASGIApp(
        sio, 
        other_asgi_app=app_fastapi,
        static_files={
            '/assets/icons/github.svg': {
                'filename':'../frontend/dist/assets/icons/github.svg',
                'content_type':'image/svg+xml'
                },
            '': '../frontend/dist',
            '/': '../frontend/dist/index.html',
        }
    )

    @app_fastapi.get("/api/hello")
    async def index():
        return {"result": "Index"}
    

    ############
    # File Uploder
    # ########## 
    UPLOAD_DIR = "upload_dir"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    MODEL_DIR = "/MMVC_Trainer/logs"
    os.makedirs(MODEL_DIR, exist_ok=True)

    @app_fastapi.post("/upload_file")
    async def post_upload_file(
        file:UploadFile = File(...), 
        filename: str = Form(...)
        ):
        return upload_file(UPLOAD_DIR, file, filename)

    @app_fastapi.post("/load_model")
    async def post_load_model(
        modelFilename: str = Form(...),
        modelFilenameChunkNum: int = Form(...),
        configFilename: str = Form(...)
        ):

        modelFilePath = concat_file_chunks(UPLOAD_DIR, modelFilename, modelFilenameChunkNum,UPLOAD_DIR)
        print(f'File saved to: {modelFilePath}')
        configFilePath = os.path.join(UPLOAD_DIR, configFilename)

        namespace.loadModel(configFilePath, modelFilePath)
        return {"load": f"{modelFilePath}, {configFilePath}"}

    @app_fastapi.post("/load_model_for_train")
    async def post_load_model_for_train(
        modelGFilename: str = Form(...),
        modelGFilenameChunkNum: int = Form(...),
        modelDFilename: str = Form(...),
        modelDFilenameChunkNum: int = Form(...),
        ):

        
        modelGFilePath = concat_file_chunks(UPLOAD_DIR, modelGFilename, modelGFilenameChunkNum, MODEL_DIR)
        modelDFilePath = concat_file_chunks(UPLOAD_DIR,  modelDFilename, modelDFilenameChunkNum,MODEL_DIR)
        return {"File saved": f"{modelGFilePath}, {modelDFilePath}"}


    @app_fastapi.post("/extract_voices")
    async def post_load_model(
        zipFilename: str = Form(...),
        zipFileChunkNum: int = Form(...),
        ):
        zipFilePath = concat_file_chunks(UPLOAD_DIR, zipFilename, zipFileChunkNum, UPLOAD_DIR)
        shutil.unpack_archive(zipFilePath, "/MMVC_Trainer/dataset/textful/")
        return {"Zip file unpacked": f"{zipFilePath}"}

    

    ############
    # Voice Changer
    # ########## 
    @app_fastapi.post("/test")
    async def post_test(voice:VoiceModel):
        try:
            # print("POST REQUEST PROCESSING....")
            gpu = voice.gpu
            srcId = voice.srcId
            dstId = voice.dstId
            timestamp = voice.timestamp
            prefixChunkSize = voice.prefixChunkSize
            buffer = voice.buffer
            wav = base64.b64decode(buffer)

            if wav==0:
                samplerate, data=read("dummy.wav")
                unpackedData = data
            else:
                unpackedData = np.array(struct.unpack('<%sh'%(len(wav) // struct.calcsize('<h') ), wav))
                write("logs/received_data.wav", 24000, unpackedData.astype(np.int16))

            changedVoice = namespace.changeVoice(gpu, srcId, dstId, timestamp, prefixChunkSize, unpackedData)

            changedVoiceBase64 = base64.b64encode(changedVoice).decode('utf-8')
            data = {
                "gpu":gpu,
                "srcId":srcId,
                "dstId":dstId,
                "timestamp":timestamp,
                "prefixChunkSize":prefixChunkSize,
                "changedVoiceBase64":changedVoiceBase64
            }

            json_compatible_item_data = jsonable_encoder(data)
            
            return JSONResponse(content=json_compatible_item_data)

        except Exception as e:
            print("REQUEST PROCESSING!!!! EXCEPTION!!!", e)
            print(traceback.format_exc())
            return str(e)


    # Trainer REST API ※ ColabがTop直下のパスにしかPOSTを投げれないようなので"REST風"
    @app_fastapi.get("/get_speakers")
    async def get_speakers():
        return mod_get_speakers()

    @app_fastapi.delete("/delete_speaker")
    async def delete_speaker(speaker:str= Form(...)):
        return mod_delete_speaker(speaker)

    @app_fastapi.get("/get_speaker_voices")
    async def get_speaker_voices(speaker:str):
        return mod_get_speaker_voices(speaker)

    @app_fastapi.get("/get_speaker_voice")
    async def get_speaker_voices(speaker:str, voice:str):
        return mod_get_speaker_voice(speaker, voice)

        
    @app_fastapi.get("/get_multi_speaker_setting")
    async def get_multi_speaker_setting():
        return mod_get_multi_speaker_setting()

    @app_fastapi.post("/post_multi_speaker_setting")
    async def post_multi_speaker_setting(setting: str = Form(...)):
        return mod_post_multi_speaker_setting(setting)

    @app_fastapi.get("/get_models")
    async def get_models():
        return mod_get_models()

    @app_fastapi.get("/get_model")
    async def get_model(model:str):
        return mod_get_model(model)

    @app_fastapi.delete("/delete_model")
    async def delete_model(model:str= Form(...)):
        return mod_delete_model(model)


    @app_fastapi.post("/post_pre_training")
    async def post_pre_training(batch:int= Form(...)):
        return mod_post_pre_training(batch)

    @app_fastapi.post("/post_start_training")
    async def post_start_training():
        print("POST START TRAINING..")
        return mod_post_start_training()

    @app_fastapi.post("/post_stop_training")
    async def post_stop_training():
        print("POST STOP TRAINING..")
        return mod_post_stop_training()

    @app_fastapi.get("/get_related_files")
    async def get_related_files():
        return mod_get_related_files()
    
    @app_fastapi.get("/get_tail_training_log")
    async def get_tail_training_log(num:int):
        return mod_get_tail_training_log(num)

    @app_fastapi.get("/get_ex_application_info")
    async def get_ex_application_info():
        json_compatible_item_data = jsonable_encoder(exApplitionInfo)
        return JSONResponse(content=json_compatible_item_data)


if __name__ == '__mp_main__':
    printMessage(f"PHASE2:{__name__}", level=2)


if __name__ == '__main__':
    printMessage(f"PHASE1:{__name__}", level=2)

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

      if os.getenv("EX_TB_PORT"):
          EX_TB_PORT = os.environ["EX_TB_PORT"]
          exApplitionInfo = int(EX_TB_PORT)
          printMessage(f"External_TeonsorBord_Port:{EX_PORT}", level=1)

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
        if args.colab == True:
          uvicorn.run(
              f"{os.path.basename(__file__)[:-3]}:app_fastapi", 
              host="0.0.0.0", 
              port=int(PORT), 
              log_level="critical"
              )
        else:
          uvicorn.run(
              f"{os.path.basename(__file__)[:-3]}:app_socketio", 
              host="0.0.0.0", 
              port=int(PORT), 
              reload=True,
              log_level="critical"
          )


