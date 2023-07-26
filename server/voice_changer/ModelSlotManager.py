import logging
from const import UPLOAD_DIR
from data.ModelSlot import ModelSlots, loadAllSlotInfo, saveSlotInfo
import json
import os
import shutil

logger = logging.getLogger("vcclient")


class ModelSlotManager:
    _instance = None

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.modelSlots = loadAllSlotInfo(self.model_dir)
        logger.info(f"[MODEL SLOT INFO] {self.modelSlots}")

    @classmethod
    def get_instance(cls, model_dir: str):
        if cls._instance is None:
            cls._instance = cls(model_dir)
        return cls._instance

    def _save_model_slot(self, slotIndex: int, slotInfo: ModelSlots):
        saveSlotInfo(self.model_dir, slotIndex, slotInfo)
        self.modelSlots = loadAllSlotInfo(self.model_dir)

    def _load_model_slot(self, slotIndex: int):
        return self.modelSlots[slotIndex]

    def getAllSlotInfo(self, reload: bool = False):
        if reload:
            self.modelSlots = loadAllSlotInfo(self.model_dir)
        return self.modelSlots

    def get_slot_info(self, slotIndex: int):
        return self._load_model_slot(slotIndex)

    def save_model_slot(self, slotIndex: int, slotInfo: ModelSlots):
        self._save_model_slot(slotIndex, slotInfo)

    def update_model_info(self, newData: str):
        print("[Voice Changer] UPDATE MODEL INFO", newData)
        newDataDict = json.loads(newData)
        slotInfo = self._load_model_slot(newDataDict["slot"])
        if newDataDict["key"] == "speakers":
            setattr(slotInfo, newDataDict["key"], json.loads(newDataDict["val"]))
        else:
            setattr(slotInfo, newDataDict["key"], newDataDict["val"])
        self._save_model_slot(newDataDict["slot"], slotInfo)

    def store_model_assets(self, params: str):
        paramsDict = json.loads(params)
        uploadPath = os.path.join(UPLOAD_DIR, paramsDict["file"])
        storeDir = os.path.join(self.model_dir, str(paramsDict["slot"]))
        storePath = os.path.join(
            storeDir,
            paramsDict["file"],
        )
        try:
            shutil.move(uploadPath, storePath)
            slotInfo = self._load_model_slot(paramsDict["slot"])
            setattr(slotInfo, paramsDict["name"], storePath)
            self._save_model_slot(paramsDict["slot"], slotInfo)
        except Exception as e:
            print("Exception::::", e)
