import json
import torch
from onnxsim import simplify
import onnx

from voice_changer.RVC.onnx.SynthesizerTrnMs256NSFsid_ONNX import (
    SynthesizerTrnMs256NSFsid_ONNX,
)
from voice_changer.RVC.onnx.SynthesizerTrnMs256NSFsid_nono_ONNX import (
    SynthesizerTrnMs256NSFsid_nono_ONNX,
)
from voice_changer.RVC.onnx.SynthesizerTrnMsNSFsidNono_webui_ONNX import (
    SynthesizerTrnMsNSFsidNono_webui_ONNX,
)
from voice_changer.RVC.onnx.SynthesizerTrnMsNSFsid_webui_ONNX import (
    SynthesizerTrnMsNSFsid_webui_ONNX,
)
from .const import RVC_MODEL_TYPE_RVC, RVC_MODEL_TYPE_WEBUI


def export2onnx(input_model, output_model, output_model_simple, is_half, metadata):
    cpt = torch.load(input_model, map_location="cpu")
    if is_half:
        dev = torch.device("cuda", index=0)
    else:
        dev = torch.device("cpu")

    if metadata["f0"] is True and metadata["modelType"] == RVC_MODEL_TYPE_RVC:
        net_g_onnx = SynthesizerTrnMs256NSFsid_ONNX(*cpt["config"], is_half=is_half)
    elif metadata["f0"] is True and metadata["modelType"] == RVC_MODEL_TYPE_WEBUI:
        net_g_onnx = SynthesizerTrnMsNSFsid_webui_ONNX(**cpt["params"], is_half=is_half)
    elif metadata["f0"] is False and metadata["modelType"] == RVC_MODEL_TYPE_RVC:
        net_g_onnx = SynthesizerTrnMs256NSFsid_nono_ONNX(*cpt["config"])
    elif metadata["f0"] is False and metadata["modelType"] == RVC_MODEL_TYPE_WEBUI:
        net_g_onnx = SynthesizerTrnMsNSFsidNono_webui_ONNX(**cpt["params"])

    net_g_onnx.eval().to(dev)
    net_g_onnx.load_state_dict(cpt["weight"], strict=False)
    if is_half:
        net_g_onnx = net_g_onnx.half()

    if is_half:
        feats = torch.HalfTensor(1, 2192, metadata["embChannels"]).to(dev)
    else:
        feats = torch.FloatTensor(1, 2192, metadata["embChannels"]).to(dev)
    p_len = torch.LongTensor([2192]).to(dev)
    sid = torch.LongTensor([0]).to(dev)

    if metadata["f0"] is True:
        pitch = torch.zeros(1, 2192, dtype=torch.int64).to(dev)
        pitchf = torch.FloatTensor(1, 2192).to(dev)
        input_names = ["feats", "p_len", "pitch", "pitchf", "sid"]
        inputs = (
            feats,
            p_len,
            pitch,
            pitchf,
            sid,
        )

    else:
        input_names = ["feats", "p_len", "sid"]
        inputs = (
            feats,
            p_len,
            sid,
        )

    output_names = [
        "audio",
    ]

    torch.onnx.export(
        net_g_onnx,
        inputs,
        output_model,
        dynamic_axes={
            "feats": [1],
            "pitch": [1],
            "pitchf": [1],
        },
        do_constant_folding=False,
        opset_version=17,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )

    model_onnx2 = onnx.load(output_model)
    model_simp, check = simplify(model_onnx2)
    meta = model_simp.metadata_props.add()
    meta.key = "metadata"
    meta.value = json.dumps(metadata)
    onnx.save(model_simp, output_model_simple)
