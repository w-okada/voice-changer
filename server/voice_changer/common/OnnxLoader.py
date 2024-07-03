import onnx
import os
from xxhash import xxh128
from utils.hasher import compute_hash

from onnx import ModelProto
from onnx.shape_inference import infer_shapes
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.transformers.fusion_utils import FusionUtils
from onnxruntime.transformers.onnx_model import OnnxModel


def load_onnx_model(fpath: str, is_half: bool) -> ModelProto:
    if is_half:
        return load_cached_fp16_model(fpath)
    return onnx.load(fpath)


def load_cached_fp16_model(fpath: str) -> ModelProto:
    hashfile = f'{fpath}.xxh128.txt'
    try:
        with open(hashfile, 'r', encoding='utf-8') as f:
            original_hash = f.read()
    except FileNotFoundError:
        original_hash = None
    fname, _ = os.path.splitext(os.path.basename(fpath))
    fp16_fpath = os.path.join(os.path.dirname(fpath), f'{fname}.fp16.onnx')
    if original_hash is None:
        model = convert_fp16(onnx.load(fpath))
        onnx.save(model, fp16_fpath)
        with open(fpath, 'rb') as f:
            computed_hash = compute_hash(f, xxh128())
        with open(hashfile, 'w', encoding='utf-8') as f:
            f.write(computed_hash)
    else:
        with open(fpath, 'rb') as f:
            computed_hash = compute_hash(f, xxh128())
        if computed_hash != original_hash:
            model = convert_fp16(onnx.load(fpath))
            onnx.save(model, fp16_fpath)
            with open(hashfile, 'w', encoding='utf-8') as f:
                f.write(computed_hash)
        else:
            model = onnx.load(fp16_fpath)
    return model


def convert_fp16(model: ModelProto) -> ModelProto:
    model = infer_shapes(model)
    model_fp16 = convert_float_to_float16(model)
    wrapped_fp16_model = OnnxModel(model_fp16)
    fusion_utils = FusionUtils(wrapped_fp16_model)
    fusion_utils.remove_cascaded_cast_nodes()
    fusion_utils.remove_useless_cast_nodes()
    wrapped_fp16_model.topological_sort()
    return wrapped_fp16_model.model
