from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import os
from const import RVC_MODEL_DIRNAME, TMP_DIR
from Downloader import download, download_no_tqdm
from ModelSample import RVCModelSample, getModelSamples
import json

from voice_changer.RVC.ModelSlot import ModelSlot
from voice_changer.RVC.ModelSlotGenerator import _setInfoByONNX, _setInfoByPytorch


def checkRvcModelExist(model_dir: str):
    rvcModelDir = os.path.join(model_dir, RVC_MODEL_DIRNAME)
    if not os.path.exists(rvcModelDir):
        return False
    return True


def downloadInitialSampleModels(
    sampleJsons: list[str], sampleModelIds: list[str], model_dir: str
):
    sampleModels = getModelSamples(sampleJsons, "RVC")
    if sampleModels is None:
        return

    downloadParams = []
    slot_count = 0
    line_num = 0
    for initSampleId in sampleModelIds:
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
        slotInfo: ModelSlot = ModelSlot()
        # sampleParams: Any = {"files": {}}

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
        slotInfo.modelFile = modelFilePath
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
            slotInfo.indexFile = indexPath
            line_num += 1
        if hasattr(sample, "icon") and sample.icon != "":
            iconPath = os.path.join(
                slotDir,
                os.path.basename(sample.icon),
            )
            downloadParams.append(
                {
                    "url": sample.icon,
                    "saveTo": iconPath,
                    "position": line_num,
                }
            )
            slotInfo.iconFile = iconPath
            line_num += 1

        slotInfo.sampleId = sample.id
        slotInfo.credit = sample.credit
        slotInfo.description = sample.description
        slotInfo.name = sample.name
        slotInfo.termsOfUseUrl = sample.termsOfUseUrl
        slotInfo.defaultTune = 0
        slotInfo.defaultIndexRatio = 1
        slotInfo.defaultProtect = 0.5
        slotInfo.isONNX = slotInfo.modelFile.endswith(".onnx")

        # この時点ではまだファイルはダウンロードされていない
        # if slotInfo.isONNX:
        #     _setInfoByONNX(slotInfo)
        # else:
        #     _setInfoByPytorch(slotInfo)

        json.dump(asdict(slotInfo), open(os.path.join(slotDir, "params.json"), "w"))
        slot_count += 1

    # ダウンロード
    print("[Voice Changer] Downloading model files...")
    with ThreadPoolExecutor() as pool:
        pool.map(download, downloadParams)

    # メタデータ作成
    print("[Voice Changer] Generating metadata...")
    for slotId in range(slot_count):
        slotDir = os.path.join(model_dir, RVC_MODEL_DIRNAME, str(slotId))
        jsonDict = json.load(open(os.path.join(slotDir, "params.json")))
        slotInfo = ModelSlot(**jsonDict)
        if slotInfo.isONNX:
            _setInfoByONNX(slotInfo)
        else:
            _setInfoByPytorch(slotInfo)
        json.dump(asdict(slotInfo), open(os.path.join(slotDir, "params.json"), "w"))


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
