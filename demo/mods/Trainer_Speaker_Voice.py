from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import os,  base64

def mod_get_speaker_voice(speaker:str, voice:str):
        wav_file = f'/MMVC_Trainer/dataset/textful/{speaker}/wav/{voice}.wav'
        text_file = f'/MMVC_Trainer/dataset/textful/{speaker}/text/{voice}.txt'
        readable_text_file = f'/MMVC_Trainer/dataset/textful/{speaker}/readable_text/{voice}.txt'

        data = {}
        if os.path.exists(wav_file):
            with open(wav_file, "rb") as f:
                wav_data = f.read()
            wav_data_base64 = base64.b64encode(wav_data).decode('utf-8')
            data["wav"] = wav_data_base64


        if os.path.exists(text_file):
            with open(text_file, "r") as f:
                text_data = f.read()
            data["text"] = text_data

        if os.path.exists(readable_text_file):
            with open(readable_text_file, "r") as f:
                text_data = f.read()
            data["readable_text"] = text_data
        json_compatible_item_data = jsonable_encoder(data)
        return JSONResponse(content=json_compatible_item_data)
