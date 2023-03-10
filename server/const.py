import os
import sys
import tempfile

ERROR_NO_ONNX_SESSION = "ERROR_NO_ONNX_SESSION"


tmpdir = tempfile.TemporaryDirectory()
# print("generate tmpdir:::",tmpdir)
SSL_KEY_DIR = os.path.join(tmpdir.name, "keys") if hasattr(sys, "_MEIPASS") else "keys"
MODEL_DIR = os.path.join(tmpdir.name, "logs") if hasattr(sys, "_MEIPASS") else "logs"
UPLOAD_DIR = os.path.join(tmpdir.name, "upload_dir") if hasattr(sys, "_MEIPASS") else "upload_dir"
NATIVE_CLIENT_FILE_WIN = os.path.join(sys._MEIPASS, "voice-changer-native-client.exe") if hasattr(sys, "_MEIPASS") else "voice-changer-native-client"
NATIVE_CLIENT_FILE_MAC = os.path.join(sys._MEIPASS, "voice-changer-native-client.app", "Contents", "MacOS",
                                      "voice-changer-native-client") if hasattr(sys, "_MEIPASS") else "voice-changer-native-client"


TMP_DIR = os.path.join(tmpdir.name, "tmp_dir") if hasattr(sys, "_MEIPASS") else "tmp_dir"
os.makedirs(TMP_DIR, exist_ok=True)


modelType = "MMVCv15"


def getModelType():
    return modelType


def setModelType(_modelType: str):
    global modelType
    modelType = _modelType


def getFrontendPath():
    if modelType == "MMVCv15":
        frontend_path = os.path.join(sys._MEIPASS, "dist_v15") if hasattr(sys, "_MEIPASS") else "../client/demo_v15/dist"
    elif modelType == "MMVCv13":
        frontend_path = os.path.join(sys._MEIPASS, "dist_v13") if hasattr(sys, "_MEIPASS") else "../client/demo_v13/dist"
    elif modelType == "so-vits-svc-40v2":
        frontend_path = os.path.join(sys._MEIPASS, "dist_v13") if hasattr(sys, "_MEIPASS") else "../client/demo_v13/dist"
    return frontend_path
