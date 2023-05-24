from dataclasses import dataclass
import json

from const import ModelType


@dataclass
class RVCModelSample:
    id: str = ""
    lang: str = ""
    tag: str = ""
    name: str = ""
    modelUrl: str = ""
    indexUrl: str = ""
    termsOfUseUrl: str = ""
    credit: str = ""
    description: str = ""

    sampleRate: int = 48000
    modelType: str = ""
    f0: bool = True


def getModelSamples(jsonFiles: list[str], modelType: ModelType):
    try:
        samples: list[RVCModelSample] = []
        for file in jsonFiles:
            with open(file, "r", encoding="utf-8") as f:
                jsonDict = json.load(f)

            modelList = jsonDict[modelType]
            if modelType == "RVC":
                for s in modelList:
                    modelSample = RVCModelSample(**s)
                    samples.append(modelSample)

            else:
                raise RuntimeError(f"Unknown model type {modelType}")
        return samples

    except Exception as e:
        print("[Voice Changer] loading sample info error:", e)
        return None
