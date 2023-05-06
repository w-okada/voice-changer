import os
import shutil
from typing import Union
from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import UploadFile, File, Form

from restapi.mods.FileUploader import upload_file, concat_file_chunks
from voice_changer.VoiceChangerManager import VoiceChangerManager

from const import MODEL_DIR, UPLOAD_DIR, ModelType
from voice_changer.utils.LoadModelParams import FilePaths, LoadModelParams

from dataclasses import fields

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


class MMVC_Rest_Fileuploader:
    def __init__(self, voiceChangerManager: VoiceChangerManager):
        self.voiceChangerManager = voiceChangerManager
        self.router = APIRouter()
        self.router.add_api_route("/info", self.get_info, methods=["GET"])
        self.router.add_api_route("/performance", self.get_performance, methods=["GET"])
        self.router.add_api_route(
            "/upload_file", self.post_upload_file, methods=["POST"]
        )
        self.router.add_api_route(
            "/concat_uploaded_file", self.post_concat_uploaded_file, methods=["POST"]
        )
        self.router.add_api_route(
            "/update_settings", self.post_update_settings, methods=["POST"]
        )
        self.router.add_api_route("/load_model", self.post_load_model, methods=["POST"])
        self.router.add_api_route("/model_type", self.post_model_type, methods=["POST"])
        self.router.add_api_route("/model_type", self.get_model_type, methods=["GET"])
        self.router.add_api_route("/onnx", self.get_onnx, methods=["GET"])
        self.router.add_api_route(
            "/merge_model", self.post_merge_models, methods=["POST"]
        )

    def post_upload_file(self, file: UploadFile = File(...), filename: str = Form(...)):
        res = upload_file(UPLOAD_DIR, file, filename)
        json_compatible_item_data = jsonable_encoder(res)
        return JSONResponse(content=json_compatible_item_data)

    def post_concat_uploaded_file(
        self, filename: str = Form(...), filenameChunkNum: int = Form(...)
    ):
        slot = 0
        res = concat_file_chunks(
            slot, UPLOAD_DIR, filename, filenameChunkNum, UPLOAD_DIR
        )
        json_compatible_item_data = jsonable_encoder(res)
        return JSONResponse(content=json_compatible_item_data)

    def get_info(self):
        info = self.voiceChangerManager.get_info()
        json_compatible_item_data = jsonable_encoder(info)
        return JSONResponse(content=json_compatible_item_data)

    def get_performance(self):
        info = self.voiceChangerManager.get_performance()
        json_compatible_item_data = jsonable_encoder(info)
        return JSONResponse(content=json_compatible_item_data)

    def post_update_settings(
        self, key: str = Form(...), val: Union[int, str, float] = Form(...)
    ):
        print("[Voice Changer] update configuration:", key, val)
        info = self.voiceChangerManager.update_settings(key, val)
        json_compatible_item_data = jsonable_encoder(info)
        return JSONResponse(content=json_compatible_item_data)

    def post_load_model(
        self,
        slot: int = Form(...),
        pyTorchModelFilename: str = Form(...),
        onnxModelFilename: str = Form(...),
        configFilename: str = Form(...),
        clusterTorchModelFilename: str = Form(...),
        featureFilename: str = Form(...),
        indexFilename: str = Form(...),
        isHalf: bool = Form(...),
        params: str = Form(...),
    ):
        files = FilePaths(
            configFilename=configFilename,
            pyTorchModelFilename=pyTorchModelFilename,
            onnxModelFilename=onnxModelFilename,
            clusterTorchModelFilename=clusterTorchModelFilename,
            featureFilename=featureFilename,
            indexFilename=indexFilename,
        )
        props: LoadModelParams = LoadModelParams(
            slot=slot, isHalf=isHalf, params=params, files=files
        )

        # Change Filepath
        for field in fields(props.files):
            key = field.name
            val = getattr(props.files, key)
            if val != "-":
                uploadPath = os.path.join(UPLOAD_DIR, val)
                storeDir = os.path.join(UPLOAD_DIR, f"{slot}")
                os.makedirs(storeDir, exist_ok=True)
                storePath = os.path.join(storeDir, val)
                shutil.move(uploadPath, storePath)
                setattr(props.files, key, storePath)
            else:
                setattr(props.files, key, None)

        info = self.voiceChangerManager.loadModel(props)
        json_compatible_item_data = jsonable_encoder(info)
        return JSONResponse(content=json_compatible_item_data)

    def post_model_type(self, modelType: ModelType = Form(...)):
        info = self.voiceChangerManager.switchModelType(modelType)
        json_compatible_item_data = jsonable_encoder(info)
        return JSONResponse(content=json_compatible_item_data)

    def get_model_type(self):
        info = self.voiceChangerManager.getModelType()
        json_compatible_item_data = jsonable_encoder(info)
        return JSONResponse(content=json_compatible_item_data)

    def get_onnx(self):
        info = self.voiceChangerManager.export2onnx()
        json_compatible_item_data = jsonable_encoder(info)
        return JSONResponse(content=json_compatible_item_data)

    def post_merge_models(self, request: str = Form(...)):
        print(request)
        info = self.voiceChangerManager.merge_models(request)
        json_compatible_item_data = jsonable_encoder(info)
        return JSONResponse(content=json_compatible_item_data)
