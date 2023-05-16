from dataclasses import dataclass
import json

from const import ModelType


@dataclass
class RVCModelSample:
    id: str = ""
    name: str = ""
    modelUrl: str = ""
    indexUrl: str = ""
    featureUrl: str = ""
    termOfUseUrl: str = ""
    credit: str = ""
    description: str = ""


def getModelSamples(jsonPath: str, modelType: ModelType):
    try:
        with open(jsonPath, "r", encoding="utf-8") as f:
            jsonDict = json.load(f)

        modelList = jsonDict[modelType]
        if modelType == "RVC":
            samples: list[RVCModelSample] = []
            for s in modelList:
                modelSample = RVCModelSample(**s)
                samples.append(modelSample)
            return samples

        else:
            raise RuntimeError(f"Unknown model type {modelType}")
    except Exception as e:
        print("[Voice Changer] loading sample info error:", e)
        return None
