import torch
import onnxruntime

try:
    import torch_directml
except ImportError:
    import voice_changer.common.deviceManager.DummyDML as torch_directml

class DeviceManager(object):
    _instance = None
    forceTensor: bool = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = torch.device('cpu')
        self.gpu_num = torch.cuda.device_count()
        self.mps_enabled: bool = (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        )
        self.dml_enabled: bool = torch_directml.is_available()

    def setDevice(self, id: int):
        self.device = self.getDevice(id)

    def getDevice(self, id: int):
        if id == -1:
            return torch.device("cpu")

        if self.gpu_num > 0 and id < self.gpu_num:
            return torch.device("cuda", index=id)
        elif self.mps_enabled:
            return torch.device("mps")
        elif self.dml_enabled:
            return torch.device(torch_directml.device(id))
        else:
            print("[Voice Changer] Device detection error, fallback to cpu")
            return torch.device("cpu")

    @staticmethod
    def listDevices():
        devCount = torch.cuda.device_count()
        gpus = [{ "id": -1, "name": "CPU" }]
        for id in range(devCount):
            name = torch.cuda.get_device_name(id)
            memory = torch.cuda.get_device_properties(id).total_memory
            gpu = {"id": id, "name": f"{id}: {name} (CUDA) ", "memory": memory}
            gpus.append(gpu)
        devCount = torch_directml.device_count()
        for id in range(devCount):
            name = torch_directml.device_name(id)
            gpu = {"id": id, "name": f"{id}: {name} (DirectML) ", "memory": 0}
            gpus.append(gpu)
        return gpus

    def getOnnxExecutionProvider(self, gpu: int):
        cpu_settings = {
            "intra_op_num_threads": 8,
            "execution_mode": onnxruntime.ExecutionMode.ORT_PARALLEL,
            "inter_op_num_threads": 8,
        }
        availableProviders = onnxruntime.get_available_providers()
        if gpu >= 0 and "ROCMExecutionProvider" in availableProviders and self.gpu_num > 0:
            return ["ROCMExecutionProvider", "CPUExecutionProvider"], [{"device_id": gpu}, cpu_settings]
        elif gpu >= 0 and "CUDAExecutionProvider" in availableProviders and self.gpu_num > 0:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], [{"device_id": gpu}, cpu_settings]
        elif gpu >= 0 and "DmlExecutionProvider" in availableProviders:
            return ["DmlExecutionProvider", "CPUExecutionProvider"], [{"device_id": gpu}, cpu_settings]
        else:
            return ["CPUExecutionProvider"], [cpu_settings]

    def setForceTensor(self, forceTensor: bool):
        self.forceTensor = forceTensor

    def halfPrecisionAvailable(self, id: int):
        if self.gpu_num == 0:
            return False
        if id < 0:
            return False
        if self.forceTensor:
            return False

        try:
            gpuName = torch.cuda.get_device_name(id).upper()
            if (
                ("16" in gpuName and "V100" not in gpuName)
                or "P40" in gpuName.upper()
                or "1070" in gpuName
                or "1080" in gpuName
            ):
                return False
        except Exception as e:
            print(e)
            return False

        cap = torch.cuda.get_device_capability(id)
        if cap[0] < 7:  # コンピューティング機能が7以上の場合half precisionが使えるとされている（が例外がある？T500とか）
            return False

        return True

    def getDeviceMemory(self, id: int):
        try:
            return torch.cuda.get_device_properties(id).total_memory
        except Exception as e:
            # except:
            print(e)
            return 0
