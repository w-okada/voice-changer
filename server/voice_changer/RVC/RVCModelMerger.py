import os

import torch
from const import UPLOAD_DIR
from voice_changer.RVC.modelMerger.MergeModel import merge_model
from voice_changer.utils.ModelMerger import ModelMerger, ModelMergerRequest
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams


class RVCModelMerger(ModelMerger):
    @classmethod
    def merge_models(cls, params: VoiceChangerParams, request: ModelMergerRequest, storeSlot: int):
        merged = merge_model(params, request)

        # いったんは、アップロードフォルダに格納する。（歴史的経緯）
        # 後続のloadmodelを呼び出すことで永続化モデルフォルダに移動させられる。
        storeDir = os.path.join(UPLOAD_DIR)
        print("[Voice Changer] store merged model to:", storeDir)
        os.makedirs(storeDir, exist_ok=True)
        storeFile = os.path.join(storeDir, "merged.pth")
        torch.save(merged, storeFile)
        return storeFile
