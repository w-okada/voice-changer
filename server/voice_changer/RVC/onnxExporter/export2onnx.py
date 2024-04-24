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
from voice_changer.RVC.onnxExporter.SynthesizerTrnMs256NSFsid_ONNX import (
    SynthesizerTrnMs256NSFsid_ONNX,
)
from voice_changer.RVC.onnxExporter.SynthesizerTrnMs256NSFsid_nono_ONNX import (
    SynthesizerTrnMs256NSFsid_nono_ONNX,
)
from voice_changer.RVC.onnxExporter.SynthesizerTrnMs768NSFsid_ONNX import (
    SynthesizerTrnMs768NSFsid_ONNX,
)
from voice_changer.RVC.onnxExporter.SynthesizerTrnMs768NSFsid_nono_ONNX import (
    SynthesizerTrnMs768NSFsid_nono_ONNX,
)
from voice_changer.RVC.onnxExporter.SynthesizerTrnMsNSFsidNono_webui_ONNX import (
    SynthesizerTrnMsNSFsidNono_webui_ONNX,
)
from voice_changer.RVC.onnxExporter.SynthesizerTrnMsNSFsid_webui_ONNX import (
    SynthesizerTrnMsNSFsid_webui_ONNX,
)
from voice_changer.VoiceChangerParamsManager import VoiceChangerParamsManager
from settings import ServerSettings

def export2onnx(gpu: int, modelSlot: RVCModelSlot):
    model_dir = ServerSettings().model_dir
    modelFile = os.path.join(model_dir, str(modelSlot.slotIndex), os.path.basename(modelSlot.modelFile))

    output_file = os.path.splitext(os.path.basename(modelFile))[0] + ".onnx"
    output_file_simple = os.path.splitext(os.path.basename(modelFile))[0] + "_simple.onnx"
    output_path = os.path.join(TMP_DIR, output_file)
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
    _export2onnx(modelFile, output_path, output_path_simple, gpu, metadata)

    return output_file_simple


def _export2onnx(input_model, output_model, output_model_simple, gpu, metadata):
    dev = DeviceManager.get_instance().getDevice(gpu)
    if dev.type == 'privateuseone':
        dev = torch.device('cpu')
    is_half = DeviceManager.get_instance().halfPrecisionAvailable(gpu)
    is_safetensors = '.safetensors' in input_model

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

    if is_half:
        print(f'[Voice Changer] Exporting to float16 on GPU')
    else:
        print(f'[Voice Changer] Exporting to float32 on CPU')

    # EnumInferenceTypesのままだとシリアライズできないのでテキスト化
    if metadata["modelType"] == EnumInferenceTypes.pyTorchRVC.value:
        net_g_onnx = SynthesizerTrnMs256NSFsid_ONNX(*data["config"], is_half=is_half)
    elif metadata["modelType"] == EnumInferenceTypes.pyTorchWebUI.value:
        net_g_onnx = SynthesizerTrnMsNSFsid_webui_ONNX(**data["params"], is_half=is_half)
    elif metadata["modelType"] == EnumInferenceTypes.pyTorchRVCNono.value:
        net_g_onnx = SynthesizerTrnMs256NSFsid_nono_ONNX(*data["config"])
    elif metadata["modelType"] == EnumInferenceTypes.pyTorchWebUINono.value:
        net_g_onnx = SynthesizerTrnMsNSFsidNono_webui_ONNX(**data["params"])
    elif metadata["modelType"] == EnumInferenceTypes.pyTorchRVCv2.value:
        net_g_onnx = SynthesizerTrnMs768NSFsid_ONNX(*data["config"], is_half=is_half)
    elif metadata["modelType"] == EnumInferenceTypes.pyTorchRVCv2Nono.value:
        net_g_onnx = SynthesizerTrnMs768NSFsid_nono_ONNX(*data["config"])
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
    if is_half:
        net_g_onnx = net_g_onnx.half()

    featsLength = 64

    if is_half:
        feats = torch.zeros((1, featsLength, metadata["embChannels"]), dtype=torch.float16, device=dev)
    else:
        feats = torch.zeros((1, featsLength, metadata["embChannels"]), dtype=torch.float32, device=dev)
    p_len = torch.tensor([featsLength], dtype=torch.int64, device=dev)
    sid = torch.tensor([0], dtype=torch.int64, device=dev)

    if metadata["f0"]:
        pitch = torch.zeros((1, featsLength), dtype=torch.int64, device=dev)
        pitchf = torch.zeros((1, featsLength), dtype=torch.float32, device=dev)
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

    onnx_model, _ = simplify(onnx.load(output_model))
    meta = onnx_model.metadata_props.add()
    meta.key = "metadata"
    meta.value = json.dumps(metadata)
    onnx.save(onnx_model, output_model_simple)
