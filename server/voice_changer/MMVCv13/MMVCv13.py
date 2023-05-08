import sys
import os

from voice_changer.utils.LoadModelParams import LoadModelParams
from voice_changer.utils.VoiceChangerModel import AudioInOut

if sys.platform.startswith("darwin"):
    baseDir = [x for x in sys.path if x.endswith("Contents/MacOS")]
    if len(baseDir) != 1:
        print("baseDir should be only one ", baseDir)
        sys.exit()
    modulePath = os.path.join(baseDir[0], "MMVC_Client_v13", "python")
    sys.path.append(modulePath)
else:
    modulePath = os.path.join("MMVC_Client_v13", "python")
    sys.path.append(modulePath)


from dataclasses import dataclass, asdict, field
import numpy as np
import torch
import onnxruntime

from symbols import symbols  # type:ignore
from models import SynthesizerTrn  # type:ignore
from voice_changer.MMVCv13.TrainerFunctions import (
    TextAudioSpeakerCollate,
    spectrogram_torch,
    load_checkpoint,
    get_hparams_from_file,
)

from Exceptions import NoModeLoadedException


@dataclass
class MMVCv13Settings:
    gpu: int = 0
    srcId: int = 0
    dstId: int = 101

    framework: str = "PyTorch"  # PyTorch or ONNX
    pyTorchModelFile: str = ""
    onnxModelFile: str = ""
    configFile: str = ""

    # ↓mutableな物だけ列挙
    intData = ["gpu", "srcId", "dstId"]
    floatData: list[str] = field(default_factory=lambda: [])
    strData = ["framework"]


class MMVCv13:
    audio_buffer: AudioInOut | None = None

    def __init__(self):
        self.settings = MMVCv13Settings()
        self.net_g = None
        self.onnx_session = None

        self.gpu_num = torch.cuda.device_count()
        self.text_norm = torch.LongTensor([0, 6, 0])

    def loadModel(self, props: LoadModelParams):
        params = props.params

        self.settings.configFile = params["files"]["mmvcv13Config"]
        self.hps = get_hparams_from_file(self.settings.configFile)

        modelFile = params["files"]["mmvcv13Models"]
        if modelFile.endswith(".onnx"):
            self.settings.pyTorchModelFile = None
            self.settings.onnxModelFile = modelFile
        else:
            self.settings.pyTorchModelFile = modelFile
            self.settings.onnxModelFile = None

        # PyTorchモデル生成
        if self.settings.pyTorchModelFile is not None:
            self.net_g = SynthesizerTrn(
                len(symbols),
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers=self.hps.data.n_speakers,
                **self.hps.model
            )
            self.net_g.eval()
            load_checkpoint(self.settings.pyTorchModelFile, self.net_g, None)

        # ONNXモデル生成
        if self.settings.onnxModelFile is not None:
            # ort_options = onnxruntime.SessionOptions()
            # ort_options.intra_op_num_threads = 8
            # ort_options.execution_mode = ort_options.ExecutionMode.ORT_PARALLEL
            # ort_options.inter_op_num_threads = 8
            providers, options = self.getOnnxExecutionProvider()
            self.onnx_session = onnxruntime.InferenceSession(
                self.settings.onnxModelFile,
                providers=providers,
                provider_options=options,
            )
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
                # providers = self.onnx_session.get_providers()
                # print("Providers:", providers)
                # if "CUDAExecutionProvider" in providers:
                #     provider_options = [{"device_id": self.settings.gpu}]
                #     self.onnx_session.set_providers(
                #         providers=["CUDAExecutionProvider"],
                #         provider_options=provider_options,
                #     )
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
            self.onnx_session.get_providers() if self.onnx_session is not None else []
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

    def _get_spec(self, audio: AudioInOut):
        spec = spectrogram_torch(
            audio,
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

        convertOffset = -1 * convertSize
        self.audio_buffer = self.audio_buffer[convertOffset:]  # 変換対象の部分だけ抽出

        audio = torch.FloatTensor(self.audio_buffer)
        audio_norm = audio.unsqueeze(0)  # unsqueeze
        spec = self._get_spec(audio_norm)
        sid = torch.LongTensor([int(self.settings.srcId)])

        data = (self.text_norm, spec, audio_norm, sid)
        data = TextAudioSpeakerCollate()([data])

        return data

    def _onnx_inference(self, data):
        if hasattr(self, "onnx_session") is False or self.onnx_session is None:
            print("[Voice Changer] No ONNX session.")
            raise NoModeLoadedException("ONNX")

        x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [x for x in data]
        sid_tgt1 = torch.LongTensor([self.settings.dstId])
        # if spec.size()[2] >= 8:
        audio1 = (
            self.onnx_session.run(
                ["audio"],
                {
                    "specs": spec.numpy(),
                    "lengths": spec_lengths.numpy(),
                    "sid_src": sid_src.numpy(),
                    "sid_tgt": sid_tgt1.numpy(),
                },
            )[0][0, 0]
            * self.hps.data.max_wav_value
        )
        return audio1

    def _pyTorch_inference(self, data):
        if hasattr(self, "net_g") is False or self.net_g is None:
            print("[Voice Changer] No pyTorch session.")
            raise NoModeLoadedException("pytorch")

        if self.settings.gpu < 0 or self.gpu_num == 0:
            dev = torch.device("cpu")
        else:
            dev = torch.device("cuda", index=self.settings.gpu)

        with torch.no_grad():
            x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [
                x.to(dev) for x in data
            ]
            sid_target = torch.LongTensor([self.settings.dstId]).to(dev)

            audio1 = (
                self.net_g.to(dev)
                .voice_conversion(
                    spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_target
                )[0, 0]
                .data
                * self.hps.data.max_wav_value
            )
            result = audio1.float().cpu().numpy()

        return result

    def inference(self, data):
        if self.isOnnx():
            audio = self._onnx_inference(data)
        else:
            audio = self._pyTorch_inference(data)
        return audio

    def __del__(self):
        del self.net_g
        del self.onnx_session
        remove_path = os.path.join("MMVC_Client_v13", "python")
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
