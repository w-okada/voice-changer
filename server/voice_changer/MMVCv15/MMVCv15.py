import sys
import os

from voice_changer.utils.LoadModelParams import LoadModelParams
from voice_changer.utils.VoiceChangerModel import AudioInOut

if sys.platform.startswith("darwin"):
    baseDir = [x for x in sys.path if x.endswith("Contents/MacOS")]
    if len(baseDir) != 1:
        print("baseDir should be only one ", baseDir)
        sys.exit()
    modulePath = os.path.join(baseDir[0], "MMVC_Client_v15", "python")
    sys.path.append(modulePath)
else:
    modulePath = os.path.join("MMVC_Client_v15", "python")
    sys.path.append(modulePath)

from dataclasses import dataclass, asdict
import numpy as np
import torch
import onnxruntime
import pyworld as pw

from models import SynthesizerTrn  # type:ignore
from voice_changer.MMVCv15.client_modules import (
    convert_continuos_f0,
    spectrogram_torch,
    get_hparams_from_file,
    load_checkpoint,
)

from Exceptions import NoModeLoadedException, ONNXInputArgumentException

providers = [
    "OpenVINOExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]


@dataclass
class MMVCv15Settings:
    gpu: int = 0
    srcId: int = 0
    dstId: int = 101

    f0Factor: float = 1.0
    f0Detector: str = "dio"  # dio or harvest

    framework: str = "PyTorch"  # PyTorch or ONNX
    pyTorchModelFile: str = ""
    onnxModelFile: str = ""
    configFile: str = ""

    # ↓mutableな物だけ列挙
    intData = ["gpu", "srcId", "dstId"]
    floatData = ["f0Factor"]
    strData = ["framework", "f0Detector"]


class MMVCv15:
    audio_buffer: AudioInOut | None = None

    def __init__(self):
        self.settings = MMVCv15Settings()
        self.net_g = None
        self.onnx_session = None

        self.gpu_num = torch.cuda.device_count()

    def loadModel(self, props: LoadModelParams):
        params = props.params

        self.settings.configFile = params["files"]["mmvcv15Config"]
        self.hps = get_hparams_from_file(self.settings.configFile)

        modelFile = params["files"]["mmvcv15Model"]
        if modelFile.endswith(".onnx"):
            self.settings.pyTorchModelFile = None
            self.settings.onnxModelFile = modelFile
        else:
            self.settings.pyTorchModelFile = modelFile
            self.settings.onnxModelFile = None

        # PyTorchモデル生成
        self.net_g = SynthesizerTrn(
            spec_channels=self.hps.data.filter_length // 2 + 1,
            segment_size=self.hps.train.segment_size // self.hps.data.hop_length,
            inter_channels=self.hps.model.inter_channels,
            hidden_channels=self.hps.model.hidden_channels,
            upsample_rates=self.hps.model.upsample_rates,
            upsample_initial_channel=self.hps.model.upsample_initial_channel,
            upsample_kernel_sizes=self.hps.model.upsample_kernel_sizes,
            n_flow=self.hps.model.n_flow,
            dec_out_channels=1,
            dec_kernel_size=7,
            n_speakers=self.hps.data.n_speakers,
            gin_channels=self.hps.model.gin_channels,
            requires_grad_pe=self.hps.requires_grad.pe,
            requires_grad_flow=self.hps.requires_grad.flow,
            requires_grad_text_enc=self.hps.requires_grad.text_enc,
            requires_grad_dec=self.hps.requires_grad.dec,
        )
        if self.settings.pyTorchModelFile is not None:
            self.net_g.eval()
            load_checkpoint(self.settings.pyTorchModelFile, self.net_g, None)

        # ONNXモデル生成
        self.onxx_input_length = 8192
        if self.settings.onnxModelFile is not None:
            providers, options = self.getOnnxExecutionProvider()
            self.onnx_session = onnxruntime.InferenceSession(
                self.settings.onnxModelFile,
                providers=providers,
                provider_options=options,
            )
            inputs_info = self.onnx_session.get_inputs()
            for i in inputs_info:
                # print("ONNX INPUT SHAPE", i.name, i.shape)
                if i.name == "sin":
                    self.onxx_input_length = i.shape[2]
        return self.get_info()

    def getOnnxExecutionProvider(self):
        if self.settings.gpu >= 0:
            return ["CUDAExecutionProvider"], [{"device_id": self.settings.gpu}]
        elif "DmlExecutionProvider" in onnxruntime.get_available_providers():
            return ["DmlExecutionProvider"], []
        else:
            return ["CPUExecutionProvider"], [
                {
                    "intra_op_num_threads": 8,
                    "execution_mode": onnxruntime.ExecutionMode.ORT_PARALLEL,
                    "inter_op_num_threads": 8,
                }
            ]

    def isOnnx(self):
        if self.settings.onnxModelFile is not None:
            return True
        else:
            return False

    def update_settings(self, key: str, val: int | float | str):
        if key in self.settings.intData:
            val = int(val)
            setattr(self.settings, key, val)
            if key == "gpu" and self.isOnnx():
                providers, options = self.getOnnxExecutionProvider()
                self.onnx_session = onnxruntime.InferenceSession(
                    self.settings.onnxModelFile,
                    providers=providers,
                    provider_options=options,
                )
                inputs_info = self.onnx_session.get_inputs()
                for i in inputs_info:
                    if i.name == "sin":
                        self.onxx_input_length = i.shape[2]
        elif key in self.settings.floatData:
            setattr(self.settings, key, float(val))
        elif key in self.settings.strData:
            setattr(self.settings, key, str(val))
        else:
            return False

        return True

    def get_info(self):
        data = asdict(self.settings)

        data["onnxExecutionProviders"] = (
            self.onnx_session.get_providers()
            if self.settings.onnxModelFile != ""
            and self.settings.onnxModelFile is not None
            else []
        )
        files = ["configFile", "pyTorchModelFile", "onnxModelFile"]
        for f in files:
            if data[f] is not None and os.path.exists(data[f]):
                data[f] = os.path.basename(data[f])
            else:
                data[f] = ""

        return data

    def get_processing_sampling_rate(self):
        if hasattr(self, "hps") is False:
            raise NoModeLoadedException("config")
        return self.hps.data.sampling_rate

    def _get_f0(self, detector: str, newData: AudioInOut):
        audio_norm_np = newData.astype(np.float64)
        if detector == "dio":
            _f0, _time = pw.dio(
                audio_norm_np, self.hps.data.sampling_rate, frame_period=5.5
            )
            f0 = pw.stonemask(audio_norm_np, _f0, _time, self.hps.data.sampling_rate)
        else:
            f0, t = pw.harvest(
                audio_norm_np,
                self.hps.data.sampling_rate,
                frame_period=5.5,
                f0_floor=71.0,
                f0_ceil=1000.0,
            )
        f0 = convert_continuos_f0(
            f0, int(audio_norm_np.shape[0] / self.hps.data.hop_length)
        )
        f0 = torch.from_numpy(f0.astype(np.float32))
        return f0

    def _get_spec(self, newData: AudioInOut):
        audio = torch.FloatTensor(newData)
        audio_norm = audio.unsqueeze(0)  # unsqueeze
        spec = spectrogram_torch(
            audio_norm,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False,
        )
        spec = torch.squeeze(spec, 0)
        return spec

    def generate_input(
        self,
        newData: AudioInOut,
        inputSize: int,
        crossfadeSize: int,
        solaSearchFrame: int = 0,
    ):
        newData = newData.astype(np.float32) / self.hps.data.max_wav_value

        if self.audio_buffer is not None:
            self.audio_buffer = np.concatenate(
                [self.audio_buffer, newData], 0
            )  # 過去のデータに連結
        else:
            self.audio_buffer = newData

        convertSize = inputSize + crossfadeSize + solaSearchFrame

        if convertSize < 8192:
            convertSize = 8192
        if convertSize % self.hps.data.hop_length != 0:  # モデルの出力のホップサイズで切り捨てが発生するので補う。
            convertSize = convertSize + (
                self.hps.data.hop_length - (convertSize % self.hps.data.hop_length)
            )

        # ONNX は固定長
        if self.settings.framework == "ONNX":
            convertSize = self.onxx_input_length

        convertOffset = -1 * convertSize
        self.audio_buffer = self.audio_buffer[convertOffset:]  # 変換対象の部分だけ抽出

        f0 = self._get_f0(self.settings.f0Detector, self.audio_buffer)  # torch
        f0 = (f0 * self.settings.f0Factor).unsqueeze(0).unsqueeze(0)
        spec = self._get_spec(self.audio_buffer)  # torch
        sid = torch.LongTensor([int(self.settings.srcId)])
        return [spec, f0, sid]

    def _onnx_inference(self, data):
        if self.settings.onnxModelFile == "" and self.settings.onnxModelFile is None:
            print("[Voice Changer] No ONNX session.")
            raise NoModeLoadedException("ONNX")

        spec, f0, sid_src = data
        spec = spec.unsqueeze(0)
        spec_lengths = torch.tensor([spec.size(2)])
        sid_tgt1 = torch.LongTensor([self.settings.dstId])
        sin, d = self.net_g.make_sin_d(f0)
        (d0, d1, d2, d3) = d
        audio1 = (
            self.onnx_session.run(
                ["audio"],
                {
                    "specs": spec.numpy(),
                    "lengths": spec_lengths.numpy(),
                    "sin": sin.numpy(),
                    "d0": d0.numpy(),
                    "d1": d1.numpy(),
                    "d2": d2.numpy(),
                    "d3": d3.numpy(),
                    "sid_src": sid_src.numpy(),
                    "sid_tgt": sid_tgt1.numpy(),
                },
            )[0][0, 0]
            * self.hps.data.max_wav_value
        )
        return audio1

    def _pyTorch_inference(self, data):
        if (
            self.settings.pyTorchModelFile == ""
            or self.settings.pyTorchModelFile is None
        ):
            print("[Voice Changer] No pyTorch session.")
            raise NoModeLoadedException("pytorch")

        if self.settings.gpu < 0 or self.gpu_num == 0:
            dev = torch.device("cpu")
        else:
            dev = torch.device("cuda", index=self.settings.gpu)

        with torch.no_grad():
            spec, f0, sid_src = data
            spec = spec.unsqueeze(0).to(dev)
            spec_lengths = torch.tensor([spec.size(2)]).to(dev)
            f0 = f0.to(dev)
            sid_src = sid_src.to(dev)
            sid_target = torch.LongTensor([self.settings.dstId]).to(dev)

            audio1 = (
                self.net_g.to(dev)
                .voice_conversion(spec, spec_lengths, f0, sid_src, sid_target)[0, 0]
                .data
                * self.hps.data.max_wav_value
            )
            result = audio1.float().cpu().numpy()
        return result

    def inference(self, data):
        try:
            if self.isOnnx():
                audio = self._onnx_inference(data)
            else:
                audio = self._pyTorch_inference(data)
            return audio
        except onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument as e:
            print(e)
            raise ONNXInputArgumentException()

    def __del__(self):
        del self.net_g
        del self.onnx_session

        remove_path = os.path.join("MMVC_Client_v15", "python")
        sys.path = [x for x in sys.path if x.endswith(remove_path) is False]

        for key in list(sys.modules):
            val = sys.modules.get(key)
            try:
                file_path = val.__file__
                if file_path.find(remove_path + os.path.sep) >= 0:
                    print("remove", key, file_path)
                    sys.modules.pop(key)
            except:  # type:ignore
                pass
