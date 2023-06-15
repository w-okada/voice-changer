# from concurrent.futures import ThreadPoolExecutor
# from dataclasses import asdict
# import os
# from const import RVC_MODEL_DIRNAME, TMP_DIR
# from Downloader import download, download_no_tqdm
# from ModelSample import RVCModelSample, getModelSamples
# import json

# from voice_changer.RVC.ModelSlot import ModelSlot
# from voice_changer.RVC.ModelSlotGenerator import _setInfoByONNX, _setInfoByPytorch


# def downloadModelFiles(sampleInfo: RVCModelSample, useIndex: bool = True):
#     downloadParams = []

#     modelPath = os.path.join(TMP_DIR, os.path.basename(sampleInfo.modelUrl))
#     downloadParams.append(
#         {
#             "url": sampleInfo.modelUrl,
#             "saveTo": modelPath,
#             "position": 0,
#         }
#     )

#     indexPath = None
#     if useIndex is True and hasattr(sampleInfo, "indexUrl") and sampleInfo.indexUrl != "":
#         print("[Voice Changer] Download sample with index.")
#         indexPath = os.path.join(TMP_DIR, os.path.basename(sampleInfo.indexUrl))
#         downloadParams.append(
#             {
#                 "url": sampleInfo.indexUrl,
#                 "saveTo": indexPath,
#                 "position": 1,
#             }
#         )

#     iconPath = None
#     if hasattr(sampleInfo, "icon") and sampleInfo.icon != "":
#         iconPath = os.path.join(TMP_DIR, os.path.basename(sampleInfo.icon))
#         downloadParams.append(
#             {
#                 "url": sampleInfo.icon,
#                 "saveTo": iconPath,
#                 "position": 2,
#             }
#         )

#     print("[Voice Changer] Downloading model files...", end="")
#     with ThreadPoolExecutor() as pool:
#         pool.map(download_no_tqdm, downloadParams)
#     print("")
#     return modelPath, indexPath, iconPath
