import os,shutil

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import HTTPException, FastAPI, UploadFile, File, Form

from restapi.mods.FileUploader import upload_file, concat_file_chunks
from voice_changer.VoiceChangerManager import VoiceChangerManager

UPLOAD_DIR = "upload_dir"
os.makedirs(UPLOAD_DIR, exist_ok=True)
MODEL_DIR = "MMVC_Trainer/logs"
os.makedirs(MODEL_DIR, exist_ok=True)

class MMVC_Rest_Fileuploader:
    def __init__(self, voiceChangerManager:VoiceChangerManager):
        self.voiceChangerManager = voiceChangerManager
        self.router = APIRouter()
        self.router.add_api_route("/info", self.get_info, methods=["GET"])
        self.router.add_api_route("/upload_file", self.post_upload_file, methods=["POST"])
        self.router.add_api_route("/concat_uploaded_file", self.post_concat_uploaded_file, methods=["POST"])
        self.router.add_api_route("/set_onnx_provider", self.post_set_onnx_provider, methods=["POST"])
        self.router.add_api_route("/load_model", self.post_load_model, methods=["POST"])
        self.router.add_api_route("/load_model_for_train", self.post_load_model_for_train, methods=["POST"])
        self.router.add_api_route("/extract_voices", self.post_extract_voices, methods=["POST"])

        self.onnx_provider=""

    def post_upload_file(self, file: UploadFile = File(...), filename: str = Form(...)):
        return upload_file(UPLOAD_DIR, file, filename)

    def post_concat_uploaded_file(self, filename: str = Form(...), filenameChunkNum: int = Form(...)):
        modelFilePath = concat_file_chunks(
            UPLOAD_DIR, filename, filenameChunkNum, UPLOAD_DIR)
        return {"concat": f"{modelFilePath}"}
    
    def post_set_onnx_provider(self, provider: str = Form(...)):
        res = self.voiceChangerManager.set_onnx_provider(provider)
        json_compatible_item_data = jsonable_encoder(res)
        return JSONResponse(content=json_compatible_item_data)

    def get_info(self):
        info = self.voiceChangerManager.get_info()
        json_compatible_item_data = jsonable_encoder(info)
        return JSONResponse(content=json_compatible_item_data)

    def post_load_model(
        self,
        pyTorchModelFilename: str = Form(...),
        onnxModelFilename: str = Form(...),
        configFilename: str = Form(...)
    ):

        pyTorchModelFilePath = os.path.join(UPLOAD_DIR, pyTorchModelFilename) if pyTorchModelFilename != "-" else None
        onnxModelFilePath = os.path.join(UPLOAD_DIR, onnxModelFilename) if onnxModelFilename != "-" else None
        configFilePath = os.path.join(UPLOAD_DIR, configFilename)

        self.voiceChangerManager.loadModel(configFilePath, pyTorchModelFilePath, onnxModelFilePath)
        return {"load": f"{configFilePath}, {pyTorchModelFilePath}, {onnxModelFilePath}"}


    def post_load_model_for_train(
        self,
        modelGFilename: str = Form(...),
        modelGFilenameChunkNum: int = Form(...),
        modelDFilename: str = Form(...),
        modelDFilenameChunkNum: int = Form(...),
    ):
        modelGFilePath = concat_file_chunks(
            UPLOAD_DIR, modelGFilename, modelGFilenameChunkNum, MODEL_DIR)
        modelDFilePath = concat_file_chunks(
            UPLOAD_DIR,  modelDFilename, modelDFilenameChunkNum, MODEL_DIR)
        return {"File saved": f"{modelGFilePath}, {modelDFilePath}"}

    def post_extract_voices(
        self,
        zipFilename: str = Form(...),
        zipFileChunkNum: int = Form(...),
    ):
        zipFilePath = concat_file_chunks(
            UPLOAD_DIR, zipFilename, zipFileChunkNum, UPLOAD_DIR)
        shutil.unpack_archive(zipFilePath, "MMVC_Trainer/dataset/textful/")
        return {"Zip file unpacked": f"{zipFilePath}"}