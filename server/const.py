import os, sys 

frontend_path = os.path.join(sys._MEIPASS, "dist") if hasattr(sys, "_MEIPASS") else "../client/demo/dist"
ERROR_NO_ONNX_SESSION = "ERROR_NO_ONNX_SESSION"


