import os

import torch
from const import UPLOAD_DIR
from voice_changer.RVC.modelMerger.MergeModel import merge_model
from voice_changer.utils.ModelMerger import ModelMerger, ModelMergerRequest


class RVCModelMerger(ModelMerger):
    @classmethod
    def merge_models(cls, request: ModelMergerRequest, storeSlot: int):
        print("[Voice Changer] MergeRequest:", request)
        merged = merge_model(request)

        # いったんは、アップロードフォルダに格納する。（歴史的経緯）
        # 後続のloadmodelを呼び出すことで永続化モデルフォルダに移動させられる。
        storeDir = os.path.join(UPLOAD_DIR, f"{storeSlot}")
        print("[Voice Changer] store merged model to:", storeDir)
        os.makedirs(storeDir, exist_ok=True)
        storeFile = os.path.join(storeDir, "merged.pth")
        torch.save(merged, storeFile)
        return storeFile

        # # loadmodelを呼び出して永続化モデルフォルダに移動させる。
        # params = {
        #     "defaultTune": req.defaultTune,
        #     "defaultIndexRatio": req.defaultIndexRatio,
        #     "defaultProtect": req.defaultProtect,
        #     "sampleId": "",
        #     "files": {"rvcModel": storeFile},
        # }
        # props: LoadModelParams = LoadModelParams(slot=targetSlot, isHalf=True, params=params)
        # self.loadModel(props)
        # self.prepareModel(targetSlot)
        # self.settings.modelSlotIndex = targetSlot
        # self.currentSlot = self.settings.modelSlotIndex
