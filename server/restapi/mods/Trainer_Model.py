
from fastapi.responses import FileResponse
import os

def mod_get_model(modelFile:str):
        modelPath = os.path.join("MMVC_Trainer/logs", modelFile)
        return FileResponse(path=modelPath)

def mod_delete_model(modelFile:str):
        modelPath = os.path.join("MMVC_Trainer/logs", modelFile)
        os.unlink(modelPath)
        return {"Model deleted": f"{modelFile}"}

