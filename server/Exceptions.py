class NoModeLoadedException(Exception):
    def __init__(self, framework):
        self.framework = framework

    def __str__(self):
        return repr(f"No model for {self.framework} loaded. Please confirm the model uploaded.")


class HalfPrecisionChangingException(Exception):
    def __str__(self):
        return repr("HalfPrecision related exception.")


class DeviceChangingException(Exception):
    def __str__(self):
        return repr("Device changing...")


class NotEnoughDataExtimateF0(Exception):
    def __str__(self):
        return repr("Not enough data to estimate f0.")


class ONNXInputArgumentException(Exception):
    def __str__(self):
        return repr("ONNX received invalid argument.")


class DeviceCannotSupportHalfPrecisionException(Exception):
    def __str__(self):
        return repr("Device cannot support half precision.")


class VoiceChangerIsNotSelectedException(Exception):
    def __str__(self):
        return repr("Voice Changer is not selected.")


class WeightDownladException(Exception):
    def __str__(self):
        return repr("Failed to download weight.")


class PipelineCreateException(Exception):
    def __str__(self):
        return repr("Failed to create Pipeline.")


class PipelineNotInitializedException(Exception):
    def __str__(self):
        return repr("Pipeline is not initialized.")
