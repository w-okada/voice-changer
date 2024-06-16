import onnx
import os
from xxhash import xxh128
from utils.hasher import compute_hash
from onnxconverter_common import float16

def load_onnx_model(fpath: str, is_half: bool) -> onnx.ModelProto:
    if is_half:
        return load_cached_fp16_model(fpath)
    return onnx.load(fpath)

def load_cached_fp16_model(fpath: str) -> onnx.ModelProto:
    hashfile = f'{fpath}.xxh128.txt'
    try:
        with open(hashfile, 'r', encoding='utf-8') as f:
            original_hash = f.read()
    except FileNotFoundError:
        original_hash = None
    fname, _ = os.path.splitext(os.path.basename(fpath))
    fp16_fpath = os.path.join(os.path.dirname(fpath), f'{fname}.fp16.onnx')
    if original_hash is None:
        model: onnx.ModelProto = float16.convert_float_to_float16(onnx.load(fpath))
        onnx.save(model, fp16_fpath)
        with open(fpath, 'rb') as f:
            computed_hash = compute_hash(f, xxh128())
        with open(hashfile, 'w', encoding='utf-8') as f:
            f.write(computed_hash)
    else:
        with open(fpath, 'rb') as f:
            computed_hash = compute_hash(f, xxh128())
        if computed_hash != original_hash:
            model: onnx.ModelProto = float16.convert_float_to_float16(onnx.load(fpath))
            onnx.save(model, fp16_fpath)
            with open(hashfile, 'w', encoding='utf-8') as f:
                f.write(computed_hash)
        else:
            model = onnx.load(fp16_fpath)
    return model
