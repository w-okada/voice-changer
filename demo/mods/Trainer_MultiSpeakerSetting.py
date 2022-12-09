from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import os

MULTI_SPEAKER_SETTING_PATH = "MMVC_Trainer/dataset/multi_speaker_correspondence.txt"
def mod_get_multi_speaker_setting():
    data = {}
    if os.path.isfile(MULTI_SPEAKER_SETTING_PATH) == False:
        with open(MULTI_SPEAKER_SETTING_PATH, "w") as f:
            f.write("")
            f.flush()
        f.close()

    with open(MULTI_SPEAKER_SETTING_PATH, "r") as f:
        setting = f.read()
        data["multi_speaker_setting"] = setting
    json_compatible_item_data = jsonable_encoder(data)
    return JSONResponse(content=json_compatible_item_data)


def mod_post_multi_speaker_setting(setting:str):
    with open(MULTI_SPEAKER_SETTING_PATH, "w") as f:
        f.write(setting)
        f.flush()
    f.close()
    return {"Write Multispeaker setting": f"{setting}"}