import os
import sys
import tempfile
from typing import Literal, TypeAlias


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
UPLOAD_DIR = (
    os.path.join(tmpdir.name, "upload_dir")
    if hasattr(sys, "_MEIPASS")
    else "upload_dir"
)
NATIVE_CLIENT_FILE_WIN = (
    os.path.join(sys._MEIPASS, "voice-changer-native-client.exe")  # type: ignore
    if hasattr(sys, "_MEIPASS")
    else "voice-changer-native-client"
)
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

HUBERT_ONNX_MODEL_PATH = (
    os.path.join(sys._MEIPASS, "model_hubert/hubert_simple.onnx")  # type: ignore
    if hasattr(sys, "_MEIPASS")
    else "model_hubert/hubert_simple.onnx"
)


TMP_DIR = (
    os.path.join(tmpdir.name, "tmp_dir") if hasattr(sys, "_MEIPASS") else "tmp_dir"
)
os.makedirs(TMP_DIR, exist_ok=True)


def getFrontendPath():
    frontend_path = (
        os.path.join(sys._MEIPASS, "dist")
        if hasattr(sys, "_MEIPASS")
        else "../client/demo/dist"
    )
    return frontend_path
