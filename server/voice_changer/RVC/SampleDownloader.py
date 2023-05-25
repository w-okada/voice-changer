from concurrent.futures import ThreadPoolExecutor
import os
from const import RVC_MODEL_DIRNAME, TMP_DIR
from Downloader import download, download_no_tqdm
from ModelSample import RVCModelSample, getModelSamples
from typing import Any
import json


def checkRvcModelExist(model_dir: str):
    rvcModelDir = os.path.join(model_dir, RVC_MODEL_DIRNAME)
    if not os.path.exists(rvcModelDir):
        return False
    return True


def downloadInitialSampleModels(sampleJsons: list[str], model_dir: str):
    sampleModelIds = [
        ("TokinaShigure_o", True),
        ("KikotoMahiro_o", False),
        ("Amitaro_o", False),
        ("Tsukuyomi-chan_o", False),
    ]
    sampleModels = getModelSamples(sampleJsons, "RVC")
    if sampleModels is None:
        return

    downloadParams = []
    slot_count = 0
    line_num = 0
    for initSampleId in sampleModelIds:
        print(initSampleId)
        # 初期サンプルをサーチ
        match = False
        for sample in sampleModels:
            if sample.id == initSampleId[0]:
                match = True
                break
        if match is False:
            print(f"[Voice Changer] initiail sample not found. {initSampleId[0]}")
            continue

        # 検出されたら、、、
        sampleParams: Any = {"files": {}}

        slotDir = os.path.join(model_dir, RVC_MODEL_DIRNAME, str(slot_count))
        os.makedirs(slotDir, exist_ok=True)
        modelFilePath = os.path.join(
            slotDir,
            os.path.basename(sample.modelUrl),
        )
        downloadParams.append(
            {
                "url": sample.modelUrl,
                "saveTo": modelFilePath,
                "position": line_num,
            }
        )
        sampleParams["files"]["rvcModel"] = modelFilePath
        line_num += 1

        if (
            initSampleId[1] is True
            and hasattr(sample, "indexUrl")
            and sample.indexUrl != ""
        ):
            indexPath = os.path.join(
                slotDir,
                os.path.basename(sample.indexUrl),
            )
            downloadParams.append(
                {
                    "url": sample.indexUrl,
                    "saveTo": indexPath,
                    "position": line_num,
                }
            )
            sampleParams["files"]["rvcIndex"] = indexPath
            line_num += 1

        sampleParams["sampleId"] = sample.id
        sampleParams["defaultTune"] = 0
        sampleParams["defaultIndexRatio"] = 1
        sampleParams["credit"] = sample.credit
        sampleParams["description"] = sample.description
        sampleParams["name"] = sample.name
        sampleParams["sampleId"] = sample.id
        sampleParams["termsOfUseUrl"] = sample.termsOfUseUrl
        sampleParams["sampleRate"] = sample.sampleRate
        sampleParams["modelType"] = sample.modelType
        sampleParams["f0"] = sample.f0

        jsonFilePath = os.path.join(slotDir, "params.json")
        json.dump(sampleParams, open(jsonFilePath, "w"))
        slot_count += 1

    print("[Voice Changer] Downloading model files...")
    with ThreadPoolExecutor() as pool:
        pool.map(download, downloadParams)


def downloadModelFiles(sampleInfo: RVCModelSample, useIndex: bool = True):
    downloadParams = []

    modelPath = os.path.join(TMP_DIR, os.path.basename(sampleInfo.modelUrl))
    downloadParams.append(
        {
            "url": sampleInfo.modelUrl,
            "saveTo": modelPath,
            "position": 0,
        }
    )

    indexPath = None
    if (
        useIndex is True
        and hasattr(sampleInfo, "indexUrl")
        and sampleInfo.indexUrl != ""
    ):
        print("[Voice Changer] Download sample with index.")
        indexPath = os.path.join(TMP_DIR, os.path.basename(sampleInfo.indexUrl))
        downloadParams.append(
            {
                "url": sampleInfo.indexUrl,
                "saveTo": indexPath,
                "position": 1,
            }
        )

    print("[Voice Changer] Downloading model files...", end="")
    with ThreadPoolExecutor() as pool:
        pool.map(download_no_tqdm, downloadParams)
    print("")
    return modelPath, indexPath
