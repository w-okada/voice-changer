import os
from const import EnumInferenceTypes
from dataclasses import asdict
import torch
import onnxruntime
import json
import safetensors

from data.ModelSlot import RVCModelSlot
from voice_changer.common.SafetensorsUtils import convert_single
from voice_changer.utils.LoadModelParams import LoadModelParams
from voice_changer.utils.ModelSlotGenerator import ModelSlotGenerator
from settings import ServerSettings, resolve_paths

class RVCModelSlotGenerator(ModelSlotGenerator):
    @classmethod
    def load_model(cls, props: LoadModelParams):
        model_dir = resolve_paths(ServerSettings()).model_dir

        slotInfo: RVCModelSlot = RVCModelSlot()
        for file in props.files:
            if file.kind == "rvcModel":
                slotInfo.modelFile = file.name
            elif file.kind == "rvcIndex":
                slotInfo.indexFile = file.name
        slotInfo.defaultTune = 0
        slotInfo.defaultFormantShift = 0
        slotInfo.defaultIndexRatio = 0
        slotInfo.defaultProtect = 0.5
        slotInfo.isONNX = slotInfo.modelFile.endswith(".onnx")
        slotInfo.name = os.path.splitext(os.path.basename(slotInfo.modelFile))[0]
        print("RVC:: slotInfo.modelFile", slotInfo.modelFile)

        # slotInfo.iconFile = "/assets/icons/noimage.png"

        modelPath = os.path.join(model_dir, str(props.slot), os.path.basename(slotInfo.modelFile))
        if slotInfo.isONNX:
            slotInfo = cls._setInfoByONNX(modelPath, slotInfo)
        else:
            slotInfo = cls._setInfoByPytorch(modelPath, slotInfo)
            if not modelPath.endswith(".safetensors"):
                convert_single(modelPath, True)
                filename, _ = os.path.splitext(os.path.basename(modelPath))
                slotInfo.modelFile = f'{filename}.safetensors'
        return slotInfo

    @classmethod
    def _setInfoByPytorch(cls, modelPath: str, slot: RVCModelSlot):
        if modelPath.endswith(".safetensors"):
            with safetensors.safe_open(modelPath, 'pt') as data:
                cpt = data.metadata()
                cpt['f0'] = int(cpt['f0'])
                cpt['config'] = json.loads(cpt['config'])
        else:
            cpt = torch.load(modelPath, map_location="cpu")
        config_len = len(cpt["config"])
        version = cpt.get("version", "v1")

        slot = RVCModelSlot(**asdict(slot))
        slot.f0 = True if cpt["f0"] == 1 else False

        if version == "voras_beta":
            slot.f0 = True if cpt["f0"] == 1 else False
            slot.modelType = EnumInferenceTypes.pyTorchVoRASbeta.value
            slot.embChannels = 768
            slot.embOutputLayer = cpt["embedder_output_layer"] if "embedder_output_layer" in cpt else 9
            slot.useFinalProj = False

            slot.embedder = cpt["embedder_name"]
            if slot.embedder.endswith("768"):
                slot.embedder = slot.embedder[:-3]

            # if slot.embedder == "hubert":
            #     slot.embedder = "hubert"
            # elif slot.embedder == "contentvec":
            #     slot.embedder = "contentvec"
            # elif slot.embedder == "hubert_jp":
            #     slot.embedder = "hubert_jp"
            else:
                raise RuntimeError("[Voice Changer][setInfoByPytorch] unknown embedder")

        elif config_len == 18:
            # Original RVC
            if version == "v1":
                slot.modelType = EnumInferenceTypes.pyTorchRVC.value if slot.f0 else EnumInferenceTypes.pyTorchRVCNono.value
                slot.embChannels = 256
                slot.embOutputLayer = 9
                slot.useFinalProj = True
                slot.embedder = "hubert_base"
                print("[Voice Changer] Official Model(pyTorch) : v1")
            else:
                slot.modelType = EnumInferenceTypes.pyTorchRVCv2.value if slot.f0 else EnumInferenceTypes.pyTorchRVCv2Nono.value
                slot.embChannels = 768
                slot.embOutputLayer = 12
                slot.useFinalProj = False
                slot.embedder = "hubert_base"
                print("[Voice Changer] Official Model(pyTorch) : v2")

        else:
            # DDPN RVC
            slot.f0 = True if cpt["f0"] == 1 else False
            slot.modelType = EnumInferenceTypes.pyTorchWebUI.value if slot.f0 else EnumInferenceTypes.pyTorchWebUINono.value
            slot.embChannels = cpt["config"][17]
            slot.embOutputLayer = cpt["embedder_output_layer"] if "embedder_output_layer" in cpt else 9
            if slot.embChannels == 256:
                slot.useFinalProj = True
            else:
                slot.useFinalProj = False

            # DDPNモデルの情報を表示
            if slot.embChannels == 256 and slot.embOutputLayer == 9 and slot.useFinalProj:
                print("[Voice Changer] DDPN Model(pyTorch) : Official v1 like")
            elif slot.embChannels == 768 and slot.embOutputLayer == 12 and slot.useFinalProj is False:
                print("[Voice Changer] DDPN Model(pyTorch): Official v2 like")
            else:
                print(f"[Voice Changer] DDPN Model(pyTorch): ch:{slot.embChannels}, L:{slot.embOutputLayer}, FP:{slot.useFinalProj}")

            slot.embedder = cpt["embedder_name"]
            if slot.embedder.endswith("768"):
                slot.embedder = slot.embedder[:-3]

            if "speaker_info" in cpt.keys():
                for k, v in cpt["speaker_info"].items():
                    slot.speakers[int(k)] = str(v)

        slot.samplingRate = cpt["config"][-1]

        del cpt

        return slot

    @classmethod
    def _setInfoByONNX(cls, modelPath: str, slot: RVCModelSlot):
        tmp_onnx_session = onnxruntime.InferenceSession(modelPath, providers=["CPUExecutionProvider"])
        modelmeta = tmp_onnx_session.get_modelmeta()
        try:
            slot = RVCModelSlot(**asdict(slot))
            metadata = json.loads(modelmeta.custom_metadata_map["metadata"])

            # slot.modelType = metadata["modelType"]
            slot.embChannels = metadata["embChannels"]

            slot.embOutputLayer = metadata["embOutputLayer"] if "embOutputLayer" in metadata else 9
            slot.useFinalProj = metadata["useFinalProj"] if "useFinalProj" in metadata else True if slot.embChannels == 256 else False

            if slot.embChannels == 256:
                slot.useFinalProj = True
            else:
                slot.useFinalProj = False

            # ONNXモデルの情報を表示
            if slot.embChannels == 256 and slot.embOutputLayer == 9 and slot.useFinalProj:
                print("[Voice Changer] ONNX Model: Official v1 like")
            elif slot.embChannels == 768 and slot.embOutputLayer == 12 and slot.useFinalProj is False:
                print("[Voice Changer] ONNX Model: Official v2 like")
            else:
                print(f"[Voice Changer] ONNX Model: ch:{slot.embChannels}, L:{slot.embOutputLayer}, FP:{slot.useFinalProj}")

            if "embedder" not in metadata:
                slot.embedder = "hubert_base"
            else:
                slot.embedder = metadata["embedder"]

            slot.f0 = metadata["f0"]
            slot.modelType = EnumInferenceTypes.onnxRVC.value if slot.f0 else EnumInferenceTypes.onnxRVCNono.value
            slot.samplingRate = metadata["samplingRate"]
            slot.deprecated = False

            if slot.embChannels == 256:
                if metadata["version"] == "2.1":
                    slot.version = "v1.1"  # 1.1はclipをonnx内部で実施. realtimeをdisable
                else:
                    slot.version = "v1"
            elif metadata["version"] == "2":
                slot.version = "v2"
            elif metadata["version"] == "2.1":  # 2.1はclipをonnx内部で実施. realtimeをdisable
                slot.version = "v2.1"
            elif metadata["version"] == "2.2":  # 2.1と同じ
                slot.version = "v2.2"
        except Exception as e:
            slot.modelType = EnumInferenceTypes.onnxRVC.value
            slot.embChannels = 256
            slot.embedder = "hubert_base"
            slot.f0 = True
            slot.samplingRate = 48000
            slot.deprecated = True

            print("[Voice Changer] setInfoByONNX", e)
            print("[Voice Changer] ############## !!!! CAUTION !!!! ####################")
            print("[Voice Changer] This onnxfie is depricated. Please regenerate onnxfile.")
            print("[Voice Changer] ############## !!!! CAUTION !!!! ####################")

        del tmp_onnx_session
        return slot
