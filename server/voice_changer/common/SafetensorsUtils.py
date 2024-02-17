import json
import os
import torch
import torch.nn
from typing import Tuple, List, Any
from safetensors.torch import _remove_duplicate_names, load_file, save_file


def load_model(model: torch.nn.Module, f: dict[str, Any], strict=True) -> Tuple[List[str], List[str]]:
    state_dict = { k: f.get_tensor(k) for k in f.keys() }
    model_state_dict = model.state_dict()
    to_removes = _remove_duplicate_names(model_state_dict, preferred_names=state_dict.keys())
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = set(missing)
    for to_remove_group in to_removes.values():
        for to_remove in to_remove_group:
            if to_remove not in missing:
                unexpected.append(to_remove)
            else:
                missing.remove(to_remove)
    if strict and (missing or unexpected):
        missing_keys = ", ".join([f'"{k}"' for k in sorted(missing)])
        unexpected_keys = ", ".join([f'"{k}"' for k in sorted(unexpected)])
        error = f"Error(s) in loading state_dict for {model.__class__.__name__}:"
        if missing:
            error += f"\n    Missing key(s) in state_dict: {missing_keys}"
        if unexpected:
            error += f"\n    Unexpected key(s) in state_dict: {unexpected_keys}"
        raise RuntimeError(error)
    return missing, unexpected


def check_file_size(sf_filename: str, pt_filename: str):
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size

    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(
            f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """
        )


def convert_file(
    pt_filename: str,
    sf_filename: str,
    discard_names: list[str] = [],
):
    metadata = {"format": "pt"}
    data: dict = torch.load(pt_filename, map_location="cpu")
    for k, v in data.items():
        if k in ['weight', 'state_dict']:
            continue
        if type(v) is dict or type(v) is list:
            metadata[k] = json.dumps(v)
        elif v is None:
            continue
        else:
            metadata[k] = str(v)
    if "state_dict" in data:
        tensors = data["state_dict"]
    elif "weight" in data:
        tensors = data['weight']
    else:
        tensors = data
    to_removes = _remove_duplicate_names(tensors, discard_names=discard_names)

    for kept_name, to_remove_group in to_removes.items():
        for to_remove in to_remove_group:
            if to_remove not in metadata:
                metadata[to_remove] = kept_name
            del tensors[to_remove]
    # Force tensors to be contiguous
    tensors = {k: v.contiguous() for k, v in tensors.items()}

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(tensors, sf_filename, metadata=metadata)
    check_file_size(sf_filename, pt_filename)
    retensors = load_file(sf_filename)
    for k in tensors:
        pt_tensor = tensors[k]
        sf_tensor = retensors[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


def convert_single(file_path: str, delprv: bool):
    pt_name = os.path.basename(file_path)
    filename, _ = os.path.splitext(pt_name)
    pt_filename = os.path.join(file_path)
    base_path = os.path.dirname(file_path)
    sf_name = f"{filename}.safetensors"
    sf_filename = os.path.join(base_path, sf_name)
    convert_file(pt_filename, sf_filename)
    if delprv:
        os.remove(pt_filename)