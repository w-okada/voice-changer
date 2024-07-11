import torch
import onnxruntime
import re
from typing import TypedDict, Literal
from enum import IntFlag

try:
    import torch_directml
except ImportError:
    import voice_changer.common.deviceManager.DummyDML as torch_directml

class CoreMLFlag(IntFlag):
    USE_CPU_ONLY = 0x001
    ENABLE_ON_SUBGRAPH = 0x002
    ONLY_ENABLE_DEVICE_WITH_ANE = 0x004
    ONLY_ALLOW_STATIC_INPUT_SHAPES = 0x008
    CREATE_MLPROGRAM = 0x010

class DevicePresentation(TypedDict):
    id: int
    name: str
    memory: int
    backend: Literal['cpu', 'cuda', 'directml', 'mps']

class DeviceManager(object):
    _instance = None

    @classmethod
    def get_instance(cls):
        # TODO: Dictionary of device manager and client sessions (?)
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = torch.device('cpu')
        self.cuda_enabled = torch.cuda.is_available()
        self.mps_enabled: bool = (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        )
        self.dml_enabled: bool = torch_directml.is_available()
        self.fp16_available = False
        self.force_fp32 = False
        print('[Voice Changer] Initialized DeviceManager. Available backends:')
        print(f'[Voice Changer] * DirectML: {self.dml_enabled}, device count: {torch_directml.device_count()}')
        print(f'[Voice Changer] * CUDA: {self.cuda_enabled}, device count: {torch.cuda.device_count()}')
        print(f'[Voice Changer] * MPS: {self.mps_enabled}')

    def initialize(self, device_id: int, force_fp32: bool):
        self.set_device(device_id)
        self.set_force_fp32(force_fp32)

    def set_device(self, id: int):
        if self.mps_enabled:
            torch.mps.empty_cache()
        elif self.cuda_enabled:
            torch.cuda.empty_cache()

        device, metadata = self._get_device(id)
        self.device = device
        self.device_metadata = metadata
        self.fp16_available = self.is_fp16_available()
        print(f'[Voice Changer] Switched to {metadata["name"]} ({device}). FP16 support: {self.fp16_available}')

    def use_fp16(self):
        return self.fp16_available and not self.force_fp32

    # TODO: This function should also accept backend type
    def _get_device(self, dev_id: int) -> tuple[torch.device, DevicePresentation]:
        if dev_id == -1:
            if self.mps_enabled:
                return (torch.device('mps'), { "id": -1, "name": "MPS", 'backend': 'mps' })
            else:
                return (torch.device("cpu"), { "id": -1, "name": "CPU", 'backend': 'cpu' })

        if self.cuda_enabled:
            name = torch.cuda.get_device_name(dev_id)
            memory = torch.cuda.get_device_properties(dev_id).total_memory
            return (torch.device("cuda", index=dev_id), {"id": dev_id, "name": f"{dev_id}: {name} (CUDA)", "memory": memory, 'backend': 'cuda'})
        elif self.dml_enabled:
            name = torch_directml.device_name(dev_id)
            return (torch.device(torch_directml.device(dev_id)), {"id": dev_id, "name": f"{dev_id}: {name} (DirectML)", "memory": 0, 'backend': 'directml'})
        raise Exception(f'Failed to find device with index {dev_id}')

    @staticmethod
    def list_devices() -> list[DevicePresentation]:
        devCount = torch.cuda.device_count()
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            devices = [{ "id": -1, "name": "MPS", 'backend': 'mps' }]
        else:
            devices = [{ "id": -1, "name": "CPU", 'backend': 'cpu' }]
        for id in range(devCount):
            name = torch.cuda.get_device_name(id)
            memory = torch.cuda.get_device_properties(id).total_memory
            device = {"id": id, "name": f"{id}: {name} (CUDA)", "memory": memory, 'backend': 'cuda'}
            devices.append(device)
        devCount = torch_directml.device_count()
        for id in range(devCount):
            name = torch_directml.device_name(id)
            device = {"id": id, "name": f"{id}: {name} (DirectML)", "memory": 0, 'backend': 'directml'}
            devices.append(device)
        return devices

    def get_onnx_execution_provider(self):
        cpu_settings = {
            "intra_op_num_threads": 8,
            "execution_mode": onnxruntime.ExecutionMode.ORT_PARALLEL,
            "inter_op_num_threads": 8,
        }
        availableProviders = onnxruntime.get_available_providers()
        if self.device.type == 'cuda' and "ROCMExecutionProvider" in availableProviders:
            return ["ROCMExecutionProvider", "CPUExecutionProvider"], [{"device_id": self.device.index}, cpu_settings]
        elif self.device.type == 'cuda' and "CUDAExecutionProvider" in availableProviders:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], [{"device_id": self.device.index}, cpu_settings]
        elif self.device.type == 'privateuseone' and "DmlExecutionProvider" in availableProviders:
            return ["DmlExecutionProvider", "CPUExecutionProvider"], [{"device_id": self.device.index}, cpu_settings]
        elif 'CoreMLExecutionProvider' in availableProviders:
            coreml_flags = CoreMLFlag.ONLY_ENABLE_DEVICE_WITH_ANE
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"], [{'coreml_flags': coreml_flags}, cpu_settings]
        else:
            return ["CPUExecutionProvider"], [cpu_settings]

    def set_force_fp32(self, force_fp32: bool):
        if self.mps_enabled:
            torch.mps.empty_cache()
        elif self.cuda_enabled:
            torch.cuda.empty_cache()
        self.force_fp32 = force_fp32

    def is_int8_avalable(self):
        if self.device.type == 'cpu':
            return True
        # TODO: Need information on INT8 support on GPUs.
        return False

    def is_fp16_available(self):
        # TODO: Maybe need to add bfloat16 support?
        # FP16 is not supported on CPU
        if self.device.type == 'cpu':
            return False

        device_name_uppercase = self.device_metadata['name'].upper()
        # TODO: Need information and filtering for Radeon and Intel GPUs
        # All Radeon GPUs starting from GCN 1 (Radeon HD 7000 series and later) reportedly have 2:1 FP16 performance
        # Intel UHD Graphics 600 and later reportedly have 2:1 FP16 performance
        # All Intel Arc GPUs reportedly have 2:1 FP16 performance or better
        ignored_nvidia_gpu = re.search(r'(GTX|RTX|TESLA|QUADRO) (V100|[789]\d{2}|1[06]\d{2}|P40|TITAN)', device_name_uppercase)
        if ignored_nvidia_gpu is not None:
            return False

        # FIXME: Apparently FP16 does not work well for Intel iGPUs in DirectML backend.
        # Causes problems with Intel UHD on 10th Gen Intel CPU.
        # TODO: To confirm if works well on Arc GPUs
        ignored_intel_gpu = 'INTEL' in device_name_uppercase and 'ARC' not in device_name_uppercase
        if ignored_intel_gpu:
            return False

        if self.device == 'cuda':
            major, _ = torch.cuda.get_device_capability(self.device)
            if major < 7:  # コンピューティング機能が7以上の場合half precisionが使えるとされている（が例外がある？T500とか）
                return False

        return True
