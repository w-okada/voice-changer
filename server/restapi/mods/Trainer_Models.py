
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from restapi.utils.files import get_file_list
import os

def mod_get_models():
    gModels = get_file_list(f'MMVC_Trainer/logs/G*.pth')
    dModels = get_file_list(f'MMVC_Trainer/logs/D*.pth')
    configs = get_file_list(f'MMVC_Trainer/logs/config.json')
    models = []
    models.extend(gModels)
    models.extend(dModels)
    models.extend(configs)
    models = [ os.path.basename(x) for x in models]

    models = sorted(models)
    data = {
            "models":models
            }
    json_compatible_item_data = jsonable_encoder(data)
    return JSONResponse(content=json_compatible_item_data)

