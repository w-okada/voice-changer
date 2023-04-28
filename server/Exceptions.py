class NoModeLoadedException(Exception):
    def __init__(self, framework):
        self.framework = framework

    def __str__(self):
        return repr(
            f"No model for {self.framework} loaded. Please confirm the model uploaded."
        )


class ONNXInputArgumentException(Exception):
    def __str__(self):
        return repr("ONNX received invalid argument.")
