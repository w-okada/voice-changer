import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Tuple

from const import RVCSampleMode, getSampleJsonAndModelIds
from data.ModelSample import ModelSamples, generateModelSample
from data.ModelSlot import RVCModelSlot
from voice_changer.ModelSlotManager import ModelSlotManager
from voice_changer.RVC.RVCModelSlotGenerator import RVCModelSlotGenerator
from downloader.Downloader import download, download_no_tqdm


def downloadInitialSamples(mode: RVCSampleMode, model_dir: str):
    sampleJsonUrls, sampleModels = getSampleJsonAndModelIds(mode)
    sampleJsons = _downloadSampleJsons(sampleJsonUrls)
    if os.path.exists(model_dir):
        print("[Voice Changer] model_dir is already exists. skip download samples.")
        return
    samples = _generateSampleList(sampleJsons)
    slotIndex = list(range(len(sampleModels)))
    _downloadSamples(samples, sampleModels, model_dir, slotIndex)

    pass


def downloadSample(mode: RVCSampleMode, modelId: str, model_dir: str, slotIndex: int, params: Any):
    sampleJsonUrls, _sampleModels = getSampleJsonAndModelIds(mode)
    sampleJsons = _generateSampleJsons(sampleJsonUrls)
    samples = _generateSampleList(sampleJsons)
    _downloadSamples(samples, [(modelId, params)], model_dir, [slotIndex], withoutTqdm=True)
    pass


def getSampleInfos(mode: RVCSampleMode):
    sampleJsonUrls, _sampleModels = getSampleJsonAndModelIds(mode)
    sampleJsons = _generateSampleJsons(sampleJsonUrls)
    samples = _generateSampleList(sampleJsons)
    return samples


def _downloadSampleJsons(sampleJsonUrls: list[str]):
    sampleJsons = []
    for url in sampleJsonUrls:
        filename = os.path.basename(url)
        download_no_tqdm({"url": url, "saveTo": filename, "position": 0})
        sampleJsons.append(filename)
    return sampleJsons


def _generateSampleJsons(sampleJsonUrls: list[str]):
    sampleJsons = []
    for url in sampleJsonUrls:
        filename = os.path.basename(url)
        sampleJsons.append(filename)
    return sampleJsons


def _generateSampleList(sampleJsons: list[str]):
    samples: list[ModelSamples] = []
    for file in sampleJsons:
        with open(file, "r", encoding="utf-8") as f:
            jsonDict = json.load(f)
        for vcType in jsonDict:
            for sampleParams in jsonDict[vcType]:
                sample = generateModelSample(sampleParams)
                samples.append(sample)
    return samples


def _downloadSamples(samples: list[ModelSamples], sampleModelIds: list[Tuple[str, Any]], model_dir: str, slotIndex: list[int], withoutTqdm=False):
    downloadParams = []
    line_num = 0
    modelSlotManager = ModelSlotManager.get_instance(model_dir)

    for i, initSampleId in enumerate(sampleModelIds):
        targetSampleId = initSampleId[0]
        targetSampleParams = initSampleId[1]
        targetSlotIndex = slotIndex[i]

        # 初期サンプルをサーチ
        match = False
        for sample in samples:
            if sample.id == targetSampleId:
                match = True
                break
        if match is False:
            print(f"[Voice Changer] initiail sample not found. {targetSampleId}")
            continue

        # 検出されたら、、、
        slotDir = os.path.join(model_dir, str(targetSlotIndex))
        if sample.voiceChangerType == "RVC":
            slotInfo: RVCModelSlot = RVCModelSlot()

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

            if targetSampleParams["useIndex"] is True and hasattr(sample, "indexUrl") and sample.indexUrl != "":
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
            slotInfo.defaultIndexRatio = 0
            slotInfo.defaultProtect = 0.5
            slotInfo.isONNX = slotInfo.modelFile.endswith(".onnx")
            modelSlotManager.save_model_slot(targetSlotIndex, slotInfo)
        else:
            print(f"[Voice Changer] {sample.voiceChangerType} is not supported.")

    # ダウンロード
    print("[Voice Changer] Downloading model files...")
    if withoutTqdm:
        with ThreadPoolExecutor() as pool:
            pool.map(download_no_tqdm, downloadParams)
    else:
        with ThreadPoolExecutor() as pool:
            pool.map(download, downloadParams)

    # メタデータ作成
    print("[Voice Changer] Generating metadata...")
    for targetSlotIndex in slotIndex:
        slotInfo = modelSlotManager.get_slot_info(targetSlotIndex)
        if slotInfo.voiceChangerType == "RVC":
            if slotInfo.isONNX:
                slotInfo = RVCModelSlotGenerator._setInfoByONNX(slotInfo)
            else:
                slotInfo = RVCModelSlotGenerator._setInfoByPytorch(slotInfo)
            
            modelSlotManager.save_model_slot(targetSlotIndex, slotInfo)
