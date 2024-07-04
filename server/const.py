from enum import Enum
import os
import sys
import tempfile
from typing import Literal, TypeAlias
import numpy as np


VoiceChangerType: TypeAlias = Literal[
    "RVC",
]

ROOT_PATH = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.realpath(sys.argv[0]))

LOG_FILE = os.path.join(ROOT_PATH, 'vcclient.log')
DOTENV_FILE = os.path.join(ROOT_PATH, '.env')
STORED_SETTING_FILE = os.path.join(ROOT_PATH, 'stored_setting.json')
ASSETS_FILE = os.path.join(ROOT_PATH, 'assets.json')

SERVER_DEVICE_SAMPLE_RATES = [16000, 32000, 44100, 48000, 96000, 192000]

tmpdir = tempfile.TemporaryDirectory()
SSL_KEY_DIR = os.path.join(tmpdir.name, "keys") if hasattr(sys, "_MEIPASS") else "keys"
UPLOAD_DIR = os.path.join(tmpdir.name, "upload_dir") if hasattr(sys, "_MEIPASS") else "upload_dir"
TMP_DIR = os.path.join(tmpdir.name, "tmp_dir") if hasattr(sys, "_MEIPASS") else "tmp_dir"

STORED_SETTINGS = {
    "version",
    "enableServerAudio", "serverAudioSampleRate", "serverInputDeviceId", "serverOutputDeviceId", "serverMonitorDeviceId", "serverInputAudioGain", "serverOutputAudioGain",
    "crossFadeOverlapSize", "protect",
    "modelSlotIndex", "serverReadChunkSize", "extraConvertSize", "gpu", "forceFp32",
    "f0Detector", "silenceFront", "silentThreshold",
}

EDITION_FILE = os.path.join(sys._MEIPASS, "edition.txt") if hasattr(sys, "_MEIPASS") else 'edition.txt'

FRONTEND_DIR = os.path.join(sys._MEIPASS, "dist") if hasattr(sys, "_MEIPASS") else "../client/demo/dist"

VERSION_FILE = os.path.join(sys._MEIPASS, "version.txt") if hasattr(sys, "_MEIPASS") else 'version.txt'

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

F0_MIN = 50
F0_MAX = 1100
F0_MEL_MIN = 1127 * np.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * np.log(1 + F0_MAX / 700)

PitchExtractorType: TypeAlias = Literal[
    "crepe_full",
    "crepe_tiny",
    "crepe_full_onnx",
    "crepe_tiny_onnx",
    "rmvpe",
    "rmvpe_onnx",
    "fcpe",
    "fcpe_onnx",
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

def get_edition():
    if not os.path.exists(EDITION_FILE):
        return '-'
    with open(EDITION_FILE, 'r') as f:
        return f.read()

def get_version():
    if not os.path.exists(VERSION_FILE):
        return 'Development'
    with open(VERSION_FILE, 'r') as f:
        return f.read()

def getSampleJsonAndModelIds(mode: RVCSampleMode):
    if mode == "production":
        return [
            "https://huggingface.co/wok000/vcclient_model/raw/main/samples_0004_t.json",
        ], [
            ("Tsukuyomi-chan_t", {"useIndex": False}),
            ("Amitaro_t", {"useIndex": False}),
            ("KikotoMahiro_t", {"useIndex": False}),
            ("TokinaShigure_t", {"useIndex": False}),
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

MAX_SLOT_NUM = 500
