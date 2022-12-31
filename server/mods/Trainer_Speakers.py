from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from trainer_mods.files import get_dir_list
import os
# CreateはFileUploaderで実装。

def mod_get_speakers():
    os.makedirs("MMVC_Trainer/dataset/textful", exist_ok=True)
    speakers = get_dir_list("MMVC_Trainer/dataset/textful/")
    
    data = {
        "speakers":sorted(speakers)
    }
    json_compatible_item_data = jsonable_encoder(data)
    return JSONResponse(content=json_compatible_item_data)
