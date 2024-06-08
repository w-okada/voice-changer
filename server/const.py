from enum import Enum
import os
import sys
import tempfile
from typing import Literal, TypeAlias


VoiceChangerType: TypeAlias = Literal[
    "RVC",
]

STORED_SETTING_FILE = "stored_setting.json"
ASSETS_FILE = 'assets.json'

SERVER_DEVICE_SAMPLE_RATES = [16000, 32000, 44100, 48000, 96000, 192000]

tmpdir = tempfile.TemporaryDirectory()
SSL_KEY_DIR = os.path.join(tmpdir.name, "keys") if hasattr(sys, "_MEIPASS") else "keys"
UPLOAD_DIR = os.path.join(tmpdir.name, "upload_dir") if hasattr(sys, "_MEIPASS") else "upload_dir"
TMP_DIR = os.path.join(tmpdir.name, "tmp_dir") if hasattr(sys, "_MEIPASS") else "tmp_dir"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

STORED_SETTINGS = {
    "version",
    "enableServerAudio", "serverAudioSampleRate", "serverInputDeviceId", "serverOutputDeviceId", "serverMonitorDeviceId", "serverInputAudioGain", "serverOutputAudioGain",
    "crossFadeOverlapSize", "protect",
    "modelSlotIndex", "serverReadChunkSize", "extraConvertSize", "gpu",
    "f0Detector", "rvcQuality", "silenceFront", "silentThreshold",
}

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
EDITION_FILE = os.path.join(sys._MEIPASS, "edition.txt") if hasattr(sys, "_MEIPASS") else 'editions/edition.txt'

ROOT_PATH = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else '.'

FRONTEND_DIR = os.path.join(sys._MEIPASS, "dist") if hasattr(sys, "_MEIPASS") else "../client/demo/dist"

EmbedderType: TypeAlias = Literal["hubert_base", "contentvec"]


class EnumInferenceTypes(Enum):
    pyTorchRVC = "pyTorchRVC"
    pyTorchRVCNono = "pyTorchRVCNono"
    pyTorchRVCv2 = "pyTorchRVCv2"
    pyTorchRVCv2Nono = "pyTorchRVCv2Nono"
    pyTorchWebUI = "pyTorchWebUI"
    pyTorchWebUINono = "pyTorchWebUINono"
    pyTorchVoRASbeta = "pyTorchVoRASbeta"
    onnxRVC = "onnxRVC"
    onnxRVCNono = "onnxRVCNono"


PitchExtractorType: TypeAlias = Literal[
    "dio",
    "harvest",
    "crepe_full",
    "crepe_tiny",
    "crepe_full_onnx",
    "crepe_tiny_onnx",
    "rmvpe",
    "rmvpe_onnx",
]

ServerAudioDeviceType: TypeAlias = Literal["audioinput", "audiooutput"]

RVCSampleMode: TypeAlias = Literal[
    "production",
    "testAll",
    "testOfficial",
    "testDDPNTorch",
    "testDDPNONNX",
    "testONNXFull",
    "",
]


def getSampleJsonAndModelIds(mode: RVCSampleMode):
    if mode == "production":
        return [
            "https://huggingface.co/wok000/vcclient_model/raw/main/samples_0004_t.json",
            "https://huggingface.co/wok000/vcclient_model/raw/main/samples_0004_o.json",
            "https://huggingface.co/wok000/vcclient_model/raw/main/samples_0004_d.json",
        ], [
            ("Tsukuyomi-chan_o", {"useIndex": False}),
            ("Amitaro_o", {"useIndex": False}),
            ("KikotoMahiro_o", {"useIndex": False}),
            ("TokinaShigure_o", {"useIndex": False}),
        ]
    elif mode == "testAll":
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
            ("test-ddpn-v1-f0-48k-l9-hubert_t", {"useIndex": False}),
            ("test-ddpn-v1-nof0-48k-l9-hubert_t", {"useIndex": False}),
            ("test-ddpn-v2-f0-40k-l12-hubert_t", {"useIndex": False}),
            ("test-ddpn-v2-nof0-40k-l12-hubert_t", {"useIndex": False}),
            ("test-ddpn-v2-f0-40k-l12-hubert_jp_t", {"useIndex": False}),
            ("test-ddpn-v2-nof0-40k-l12-hubert_jp_t", {"useIndex": False}),
            ("test-ddpn-v1-f0-48k-l9-hubert_o", {"useIndex": False}),
            ("test-ddpn-v1-nof0-48k-l9-hubert_o", {"useIndex": False}),
            ("test-ddpn-v2-f0-40k-l12-hubert_o", {"useIndex": False}),
            ("test-ddpn-v2-nof0-40k-l12-hubert_o", {"useIndex": False}),
            ("test-ddpn-v2-f0-40k-l12-hubert_jp_o", {"useIndex": False}),
            ("test-ddpn-v2-nof0-40k-l12-hubert_jp_o", {"useIndex": False}),
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
MAX_SLOT_NUM = 500
