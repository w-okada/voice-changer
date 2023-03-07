import os
import sys
import tempfile

# MODEL_TYPE = "MMVCv13"
MODEL_TYPE = "MMVCv15"
# MODEL_TYPE = "sovt-vits-svc"

if MODEL_TYPE == "MMVCv15":
    frontend_path = os.path.join(sys._MEIPASS, "dist") if hasattr(sys, "_MEIPASS") else "../client/demo_v15/dist"
elif MODEL_TYPE == "MMVCv13":
    frontend_path = os.path.join(sys._MEIPASS, "dist") if hasattr(sys, "_MEIPASS") else "../client/demo_v13/dist"

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

# SSL_KEY_DIR  = os.path.join(sys._MEIPASS, "keys") if hasattr(sys, "_MEIPASS") else "keys"
# MODEL_DIR = os.path.join(sys._MEIPASS, "logs") if hasattr(sys, "_MEIPASS") else "logs"
# UPLOAD_DIR = os.path.join(sys._MEIPASS, "upload_dir") if hasattr(sys, "_MEIPASS") else "upload_dir"
