import os

from fastapi import APIRouter,Form
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


from restapi.mods.Trainer_Speakers import mod_get_speakers
from restapi.mods.Trainer_Training import mod_post_pre_training, mod_post_start_training, mod_post_stop_training, mod_get_related_files, mod_get_tail_training_log
from restapi.mods.Trainer_Model import mod_get_model, mod_delete_model

from restapi.mods.Trainer_Models import mod_get_models
from restapi.mods.Trainer_MultiSpeakerSetting import mod_get_multi_speaker_setting, mod_post_multi_speaker_setting
from restapi.mods.Trainer_Speaker_Voice import mod_get_speaker_voice
from restapi.mods.Trainer_Speaker_Voices import mod_get_speaker_voices

from restapi.mods.Trainer_Speaker import mod_delete_speaker
from dataclasses import dataclass

INFO_DIR = "info"
os.makedirs(INFO_DIR, exist_ok=True)

@dataclass
class ExApplicationInfo():
    external_tensorboard_port: int

exApplitionInfo = ExApplicationInfo(external_tensorboard_port=0)

class MMVC_Rest_Trainer:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/get_speakers", self.get_speakers, methods=["GET"])
        self.router.add_api_route("/delete_speaker", self.delete_speaker, methods=["DELETE"])
        self.router.add_api_route("/get_speaker_voices", self.get_speaker_voices, methods=["GET"])
        self.router.add_api_route("/get_speaker_voice", self.get_speaker_voice, methods=["GET"])
        self.router.add_api_route("/get_multi_speaker_setting", self.get_multi_speaker_setting, methods=["GET"])
        self.router.add_api_route("/post_multi_speaker_setting", self.post_multi_speaker_setting, methods=["POST"])
        self.router.add_api_route("/get_models", self.get_models, methods=["GET"])
        self.router.add_api_route("/get_model", self.get_model, methods=["GET"])
        self.router.add_api_route("/delete_model", self.delete_model, methods=["DELETE"])
        self.router.add_api_route("/post_pre_training", self.post_pre_training, methods=["POST"])
        self.router.add_api_route("/post_start_training", self.post_start_training, methods=["POST"])
        self.router.add_api_route("/post_stop_training", self.post_stop_training, methods=["POST"])
        self.router.add_api_route("/get_related_files", self.get_related_files, methods=["GET"])
        self.router.add_api_route("/get_tail_training_log", self.get_tail_training_log, methods=["GET"])
        self.router.add_api_route("/get_ex_application_info", self.get_ex_application_info, methods=["GET"])

    def get_speakers(self):
        return mod_get_speakers()

    def delete_speaker(self, speaker: str = Form(...)):
        return mod_delete_speaker(speaker)

    def get_speaker_voices(self, speaker: str):
        return mod_get_speaker_voices(speaker)

    def get_speaker_voice(self, speaker: str, voice: str):
        return mod_get_speaker_voice(speaker, voice)

    def get_multi_speaker_setting(self):
        return mod_get_multi_speaker_setting()

    def post_multi_speaker_setting(self, setting: str = Form(...)):
        return mod_post_multi_speaker_setting(setting)

    def get_models(self):
        return mod_get_models()

    def get_model(self, model: str):
        return mod_get_model(model)

    def delete_model(self, model: str = Form(...)):
        return mod_delete_model(model)

    def post_pre_training(self, batch: int = Form(...)):
        return mod_post_pre_training(batch)

    def post_start_training(self, enable_finetuning: bool = Form(...),GModel: str = Form(...),DModel: str = Form(...)):
        print("POST START TRAINING..")
        return mod_post_start_training(enable_finetuning, GModel, DModel)

    def post_stop_training(self):
        print("POST STOP TRAINING..")
        return mod_post_stop_training()

    def get_related_files(self):
        return mod_get_related_files()

    def get_tail_training_log(self, num: int):
        return mod_get_tail_training_log(num)

    def get_ex_application_info(self):
        json_compatible_item_data = jsonable_encoder(exApplitionInfo)
        return JSONResponse(content=json_compatible_item_data)
