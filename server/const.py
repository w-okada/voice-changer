from enum import Enum
import os
import sys
import tempfile
from typing import Literal, TypeAlias


VoiceChangerType: TypeAlias = Literal[
    "MMVCv13",
    "MMVCv15",
    "so-vits-svc-40",
    "DDSP-SVC",
    "RVC",
]

ModelType: TypeAlias = Literal[
    "MMVCv15",
    "MMVCv13",
    "so-vits-svc-40v2",
    "so-vits-svc-40",
    "so-vits-svc-40_c",
    "DDSP-SVC",
    "RVC",
]

ERROR_NO_ONNX_SESSION = "ERROR_NO_ONNX_SESSION"


tmpdir = tempfile.TemporaryDirectory()
# print("generate tmpdir:::",tmpdir)
SSL_KEY_DIR = os.path.join(tmpdir.name, "keys") if hasattr(sys, "_MEIPASS") else "keys"
MODEL_DIR = os.path.join(tmpdir.name, "logs") if hasattr(sys, "_MEIPASS") else "logs"
UPLOAD_DIR = os.path.join(tmpdir.name, "upload_dir") if hasattr(sys, "_MEIPASS") else "upload_dir"
NATIVE_CLIENT_FILE_WIN = os.path.join(sys._MEIPASS, "voice-changer-native-client.exe") if hasattr(sys, "_MEIPASS") else "voice-changer-native-client"  # type: ignore
NATIVE_CLIENT_FILE_MAC = (
    os.path.join(
        sys._MEIPASS,  # type: ignore
        "voice-changer-native-client.app",
        "Contents",
        "MacOS",
        "voice-changer-native-client",
    )
    if hasattr(sys, "_MEIPASS")
    else "voice-changer-native-client"
)

HUBERT_ONNX_MODEL_PATH = os.path.join(sys._MEIPASS, "model_hubert/hubert_simple.onnx") if hasattr(sys, "_MEIPASS") else "model_hubert/hubert_simple.onnx"  # type: ignore


TMP_DIR = os.path.join(tmpdir.name, "tmp_dir") if hasattr(sys, "_MEIPASS") else "tmp_dir"
os.makedirs(TMP_DIR, exist_ok=True)


def getFrontendPath():
    frontend_path = os.path.join(sys._MEIPASS, "dist") if hasattr(sys, "_MEIPASS") else "../client/demo/dist"
    return frontend_path


# "hubert_base",  "contentvec",  "distilhubert"
class EnumEmbedderTypes(Enum):
    hubert = "hubert_base"
    contentvec = "contentvec"
    hubert_jp = "hubert-base-japanese"


class EnumInferenceTypes(Enum):
    pyTorchRVC = "pyTorchRVC"
    pyTorchRVCNono = "pyTorchRVCNono"
    pyTorchRVCv2 = "pyTorchRVCv2"
    pyTorchRVCv2Nono = "pyTorchRVCv2Nono"
    pyTorchWebUI = "pyTorchWebUI"
    pyTorchWebUINono = "pyTorchWebUINono"
    onnxRVC = "onnxRVC"
    onnxRVCNono = "onnxRVCNono"


class EnumPitchExtractorTypes(Enum):
    harvest = "harvest"
    dio = "dio"
    crepe = "crepe"


class EnumFrameworkTypes(Enum):
    pyTorch = "pyTorch"
    onnx = "onnx"


class ServerAudioDeviceTypes(Enum):
    audioinput = "audioinput"
    audiooutput = "audiooutput"


RVCSampleMode: TypeAlias = Literal[
    "production",
    "testOfficial",
    "testDDPNTorch",
    "testDDPNONNX",
    "testONNXFull",
]


def getSampleJsonAndModelIds(mode: RVCSampleMode):
    if mode == "production":
        return [
            # "https://huggingface.co/wok000/vcclient_model/raw/main/samples_0001.json",
            # "https://huggingface.co/wok000/vcclient_model/raw/main/samples_0002.json",
            "https://huggingface.co/wok000/vcclient_model/raw/main/samples_0003_t2.json",
            "https://huggingface.co/wok000/vcclient_model/raw/main/samples_0003_o2.json",
        ], [
            # ("TokinaShigure_o", {"useIndex": True}),
            # ("KikotoMahiro_o", {"useIndex": False}),
            # ("Amitaro_o", {"useIndex": False}),
            ("Tsukuyomi-chan_o", {"useIndex": False}),
        ]
    elif mode == "testOfficial":
        return [
            "https://huggingface.co/wok000/vcclient_model/raw/main/test/test_official_v1_v2.json",
            "https://huggingface.co/wok000/vcclient_model/raw/main/test/test_ddpn_v1_v2.json",
        ], [
            ("test-official-v1-f0-48k-l9-hubert_t", {"useIndex": True}),
            ("test-official-v1-nof0-48k-l9-hubert_t", {"useIndex": False}),
            ("test-official-v2-f0-40k-l12-hubert_t", {"useIndex": False}),
            ("test-official-v2-nof0-40k-l12-hubert_t", {"useIndex": False}),
            ("test-official-v1-f0-48k-l9-hubert_o", {"useIndex": True}),
            ("test-official-v1-nof0-48k-l9-hubert_o", {"useIndex": False}),
            ("test-official-v2-f0-40k-l12-hubert_o", {"useIndex": False}),
            ("test-official-v2-nof0-40k-l12-hubert_o", {"useIndex": False}),
        ]
    elif mode == "testDDPNTorch":
        return [
            "https://huggingface.co/wok000/vcclient_model/raw/main/test/test_official_v1_v2.json",
            "https://huggingface.co/wok000/vcclient_model/raw/main/test/test_ddpn_v1_v2.json",
        ], [
            ("test-ddpn-v1-f0-48k-l9-hubert_t", {"useIndex": False}),
            ("test-ddpn-v1-nof0-48k-l9-hubert_t", {"useIndex": False}),
            ("test-ddpn-v2-f0-40k-l12-hubert_t", {"useIndex": False}),
            ("test-ddpn-v2-nof0-40k-l12-hubert_t", {"useIndex": False}),
            ("test-ddpn-v2-f0-40k-l12-hubert_jp_t", {"useIndex": False}),
            ("test-ddpn-v2-nof0-40k-l12-hubert_jp_t", {"useIndex": False}),
        ]
    elif mode == "testDDPNONNX":
        return [
            "https://huggingface.co/wok000/vcclient_model/raw/main/test/test_official_v1_v2.json",
            "https://huggingface.co/wok000/vcclient_model/raw/main/test/test_ddpn_v1_v2.json",
        ], [
            ("test-ddpn-v1-f0-48k-l9-hubert_o", {"useIndex": False}),
            ("test-ddpn-v1-nof0-48k-l9-hubert_o", {"useIndex": False}),
            ("test-ddpn-v2-f0-40k-l12-hubert_o", {"useIndex": False}),
            ("test-ddpn-v2-nof0-40k-l12-hubert_o", {"useIndex": False}),
            ("test-ddpn-v2-f0-40k-l12-hubert_jp_o", {"useIndex": False}),
            ("test-ddpn-v2-nof0-40k-l12-hubert_jp_o", {"useIndex": False}),
        ]
    elif mode == "testONNXFull":
        return [
            "https://huggingface.co/wok000/vcclient_model/raw/main/test/test_official_v1_v2.json",
            "https://huggingface.co/wok000/vcclient_model/raw/main/test/test_ddpn_v1_v2.json",
        ], [
            ("test-official-v1-f0-48k-l9-hubert_o_full", {"useIndex": False}),
            ("test-official-v1-nof0-48k-l9-hubert_o_full", {"useIndex": False}),
            ("test-official-v2-f0-40k-l12-hubert_o_full", {"useIndex": False}),
            ("test-official-v2-nof0-40k-l12-hubert_o_full", {"useIndex": False}),
            ("test-ddpn-v1-f0-48k-l9-hubert_o_full", {"useIndex": False}),
            ("test-ddpn-v1-nof0-48k-l9-hubert_o_full", {"useIndex": False}),
            ("test-ddpn-v2-f0-40k-l12-hubert_o_full", {"useIndex": False}),
            ("test-ddpn-v2-nof0-40k-l12-hubert_o_full", {"useIndex": False}),
            ("test-ddpn-v2-f0-40k-l12-hubert_jp_o_full", {"useIndex": False}),
            ("test-ddpn-v2-nof0-40k-l12-hubert_jp_o_full", {"useIndex": False}),
        ]
    else:
        return [], []


RVC_MODEL_DIRNAME = "rvc"
MAX_SLOT_NUM = 10
