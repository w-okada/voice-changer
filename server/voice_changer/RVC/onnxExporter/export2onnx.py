import os
import json
import torch
from onnxsim import simplify
import onnx
import safetensors
from const import TMP_DIR, EnumInferenceTypes
from data.ModelSlot import RVCModelSlot
from voice_changer.common.SafetensorsUtils import load_model
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from ..inferencer.rvc_models.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM  # type: ignore
from settings import ServerSettings, resolve_paths
from io import BytesIO

def export2onnx(modelSlot: RVCModelSlot):
    model_dir = resolve_paths(ServerSettings()).model_dir
    modelFile = os.path.join(model_dir, str(modelSlot.slotIndex), os.path.basename(modelSlot.modelFile))

    output_file_simple = os.path.splitext(os.path.basename(modelFile))[0] + "_simple.onnx"
    output_path_simple = os.path.join(TMP_DIR, output_file_simple)
    metadata = {
        "application": "VC_CLIENT",
        "version": "2.1",
        "modelType": modelSlot.modelType,
        "samplingRate": modelSlot.samplingRate,
        "f0": modelSlot.f0,
        "embChannels": modelSlot.embChannels,
        "embedder": modelSlot.embedder,
        "embOutputLayer": modelSlot.embOutputLayer,
        "useFinalProj": modelSlot.useFinalProj,
    }

    print("[Voice Changer] Exporting onnx...")
    _export2onnx(modelFile, output_path_simple, metadata)

    return output_file_simple


def _export2onnx(input_model: str, output_model_simple: str, metadata: dict):
    device_manager = DeviceManager.get_instance()
    dev = device_manager.device
    if dev.type == 'privateuseone':
        dev = torch.device('cpu')
    is_half = False
    is_safetensors = input_model.endswith('.safetensors')

    if is_safetensors:
        cpt = safetensors.safe_open(input_model, 'pt', device=str(dev))
        m = cpt.metadata()
        data = {
            'config': json.loads(m.get('config', '{}')),
            'params': json.loads(m.get('params', '{}'))
        }
    else:
        cpt = torch.load(input_model, map_location=dev)
        data = {
            'config': cpt['config'],
            'params': cpt['params']
        }

    print(f'[Voice Changer] Exporting to ONNX on {dev}')

    # EnumInferenceTypesのままだとシリアライズできないのでテキスト化
    if metadata["modelType"] == EnumInferenceTypes.pyTorchRVC.value:
        net_g_onnx = SynthesizerTrnMsNSFsidM(*data["config"], 256, is_half=is_half)
    elif metadata["modelType"] == EnumInferenceTypes.pyTorchRVCNono.value:
        net_g_onnx = SynthesizerTrnMsNSFsidM(*data["config"], 256, is_half=is_half)
    # pyTorchWebUI already includes number of text embeddings in the model config (?)
    elif metadata["modelType"] == EnumInferenceTypes.pyTorchWebUI.value:
        net_g_onnx = SynthesizerTrnMsNSFsidM(**data["params"], is_half=is_half)
    elif metadata["modelType"] == EnumInferenceTypes.pyTorchWebUINono.value:
        net_g_onnx = SynthesizerTrnMsNSFsidM(**data["params"], is_half=is_half)
    elif metadata["modelType"] == EnumInferenceTypes.pyTorchRVCv2.value:
        net_g_onnx = SynthesizerTrnMsNSFsidM(*data["config"], 768, is_half=is_half)
    elif metadata["modelType"] == EnumInferenceTypes.pyTorchRVCv2Nono.value:
        net_g_onnx = SynthesizerTrnMsNSFsidM(*data["config"], 768, is_half=is_half)
    else:
        print(
            "unknwon::::: ",
            metadata["modelType"],
            EnumInferenceTypes.pyTorchRVCv2.value,
        )
        return

    net_g_onnx.eval().to(dev)
    if is_safetensors:
        load_model(net_g_onnx, cpt, strict=False)
    else:
        net_g_onnx.load_state_dict(cpt["weight"], strict=False)

    featsLength = 64

    feats = torch.zeros((1, featsLength, metadata["embChannels"]), dtype=torch.float32, device=dev)
    p_len = torch.tensor([featsLength], dtype=torch.int64, device=dev)
    sid = torch.tensor([0], dtype=torch.int64, device=dev)
    skip_head = torch.tensor(32, dtype=torch.int64, device=dev)
    return_length = torch.tensor(32, dtype=torch.int64, device=dev)
    formant_length = torch.tensor(25, dtype=torch.int64, device=dev)

    if metadata["f0"]:
        pitch = torch.zeros((1, featsLength), dtype=torch.int64, device=dev)
        pitchf = torch.zeros((1, featsLength), dtype=torch.float32, device=dev)
        input_names = ["feats", "p_len", "pitch", "pitchf", "sid", "skip_head", "return_length", "formant_length"]
        inputs = (
            feats,
            p_len,
            pitch,
            pitchf,
            sid,
            skip_head,
            return_length,
            formant_length,
        )

    else:
        input_names = ["feats", "p_len", "sid", "skip_head", "return_length", "formant_length"]
        inputs = (
            feats,
            p_len,
            sid,
            skip_head,
            return_length,
            formant_length,
        )

    output_names = [
        "audio",
    ]

    with BytesIO() as io:
        torch.onnx.export(
            net_g_onnx,
            inputs,
            io,
            dynamic_axes={
                "feats": [1],
                "pitch": [1],
                "pitchf": [1],
            },
            do_constant_folding=True,
            opset_version=17,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
        )
        onnx_model, _ = simplify(onnx.load_model_from_string(io.getvalue()))

    meta = onnx_model.metadata_props.add()
    meta.key = "metadata"
    meta.value = json.dumps(metadata)
    onnx.save(onnx_model, output_model_simple)
