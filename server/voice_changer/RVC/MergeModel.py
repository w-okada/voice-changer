from typing import Dict, Any
from voice_changer.RVC.MergeModelRequest import MergeModelRequest
from collections import OrderedDict
import torch
import tqdm


def merge_model(request: MergeModelRequest):
    def extract(ckpt: Dict[str, Any]):
        a = ckpt["model"]
        opt: Dict[str, Any] = OrderedDict()

        opt["weight"] = {}
        for key in a.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = a[key]
        return opt

    def load_weight(path: str):
        print(f"Loading {path}...")
        state_dict = torch.load(path, map_location="cpu")
        if "model" in state_dict:
            weight = extract(state_dict)
        else:
            weight = state_dict["weight"]
        return weight, state_dict

    files = request.files
    if len(files) == 0:
        print("no merge file..............")
        raise RuntimeError("no merge file..............")

    weights = []
    alphas = []
    for f in files:
        strength = f.strength
        if strength == 0:
            continue
        weight, state_dict = load_weight(f.filename)
        weights.append(weight)
        alphas.append(f.strength)

    alphas = [x / sum(alphas) for x in alphas]

    for weight in weights:
        if sorted(list(weight.keys())) != sorted(list(weights[0].keys())):
            raise RuntimeError("Failed to merge models.")

    merged: Dict[str, Any] = OrderedDict()
    merged["weight"] = {}
    for key in tqdm.tqdm(weights[0].keys()):
        merged["weight"][key] = 0
        for i, weight in enumerate(weights):
            merged["weight"][key] += weight[key] * alphas[i]

    merged["config"] = state_dict["config"]
    merged["params"] = state_dict["params"] if "params" in state_dict else None
    merged["sr"] = state_dict["sr"]
    merged["f0"] = state_dict["f0"]
    merged["info"] = state_dict["info"]
    merged["embedder_name"] = (
        state_dict["embedder_name"] if "embedder_name" in state_dict else None
    )
    return merged
