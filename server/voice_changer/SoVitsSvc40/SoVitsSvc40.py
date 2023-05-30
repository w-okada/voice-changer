import sys
import os

from voice_changer.utils.LoadModelParams import LoadModelParams
from voice_changer.utils.VoiceChangerModel import AudioInOut
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams

if sys.platform.startswith("darwin"):
    baseDir = [x for x in sys.path if x.endswith("Contents/MacOS")]
    if len(baseDir) != 1:
        print("baseDir should be only one ", baseDir)
        sys.exit()
    modulePath = os.path.join(baseDir[0], "so-vits-svc-40")
    sys.path.append(modulePath)
else:
    sys.path.append("so-vits-svc-40")

import io
from dataclasses import dataclass, asdict, field
import numpy as np
import torch
import onnxruntime

# onnxruntime.set_default_logger_severity(3)

import pyworld as pw

from models import SynthesizerTrn  # type:ignore
import cluster  # type:ignore
import utils
from fairseq import checkpoint_utils
import librosa

from Exceptions import NoModeLoadedException


providers = [
    "OpenVINOExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]


@dataclass
class SoVitsSvc40Settings:
    gpu: int = 0
    dstId: int = 0

    f0Detector: str = "harvest"  # dio or harvest
    tran: int = 20
    noiseScale: float = 0.3
    predictF0: int = 0  # 0:False, 1:True
    silentThreshold: float = 0.00001
    extraConvertSize: int = 1024 * 32
    clusterInferRatio: float = 0.1

    framework: str = "PyTorch"  # PyTorch or ONNX
    pyTorchModelFile: str | None = ""
    onnxModelFile: str | None = ""
    configFile: str = ""

    speakers: dict[str, int] = field(default_factory=lambda: {})

    # ↓mutableな物だけ列挙
    intData = ["gpu", "dstId", "tran", "predictF0", "extraConvertSize"]
    floatData = ["noiseScale", "silentThreshold", "clusterInferRatio"]
    strData = ["framework", "f0Detector"]


class SoVitsSvc40:
    audio_buffer: AudioInOut | None = None

    def __init__(self, params: VoiceChangerParams):
        self.settings = SoVitsSvc40Settings()
        self.net_g = None
        self.onnx_session = None

        self.raw_path = io.BytesIO()
        self.gpu_num = torch.cuda.device_count()
        self.prevVol = 0
        self.params = params
        print("so-vits-svc40 initialization:", params)

    # def loadModel(self, config: str, pyTorch_model_file: str = None, onnx_model_file: str = None, clusterTorchModel: str = None):
    def loadModel(self, props: LoadModelParams):
        params = props.params
        self.settings.configFile = params["files"]["soVitsSvc40Config"]
        self.hps = utils.get_hparams_from_file(self.settings.configFile)
        self.settings.speakers = self.hps.spk

        modelFile = params["files"]["soVitsSvc40Model"]
        if modelFile.endswith(".onnx"):
            self.settings.pyTorchModelFile = None
            self.settings.onnxModelFile = modelFile
        else:
            self.settings.pyTorchModelFile = modelFile
            self.settings.onnxModelFile = None

        clusterTorchModel = (
            params["files"]["soVitsSvc40Cluster"]
            if "soVitsSvc40Cluster" in params["files"]
            else None
        )

        content_vec_path = self.params.content_vec_500
        content_vec_onnx_path = self.params.content_vec_500_onnx
        content_vec_onnx_on = self.params.content_vec_500_onnx_on
        hubert_base_path = self.params.hubert_base

        # hubert model
        try:
            if os.path.exists(content_vec_path) is False:
                content_vec_path = hubert_base_path

            if content_vec_onnx_on is True:
                providers, options = self.getOnnxExecutionProvider()
                self.content_vec_onnx = onnxruntime.InferenceSession(
                    content_vec_onnx_path,
                    providers=providers,
                    provider_options=options,
                )
            else:
                models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
                    [content_vec_path],
                    suffix="",
                )
                model = models[0]
                model.eval()
                self.hubert_model = model.cpu()
        except Exception as e:
            print("EXCEPTION during loading hubert/contentvec model", e)

        # cluster
        try:
            if clusterTorchModel is not None and os.path.exists(clusterTorchModel):
                self.cluster_model = cluster.get_cluster_model(clusterTorchModel)
            else:
                self.cluster_model = None
        except Exception as e:
            print("EXCEPTION during loading cluster model ", e)

        # PyTorchモデル生成
        if self.settings.pyTorchModelFile is not None:
            net_g = SynthesizerTrn(
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                **self.hps.model,
            )
            net_g.eval()
            self.net_g = net_g
            utils.load_checkpoint(self.settings.pyTorchModelFile, self.net_g, None)

        # ONNXモデル生成
        if self.settings.onnxModelFile is not None:
            providers, options = self.getOnnxExecutionProvider()
            self.onnx_session = onnxruntime.InferenceSession(
                self.settings.onnxModelFile,
                providers=providers,
                provider_options=options,
            )
        return self.get_info()

    def getOnnxExecutionProvider(self):
        availableProviders = onnxruntime.get_available_providers()
        if self.settings.gpu >= 0 and "CUDAExecutionProvider" in availableProviders:
            return ["CUDAExecutionProvider"], [{"device_id": self.settings.gpu}]
        elif self.settings.gpu >= 0 and "DmlExecutionProvider" in availableProviders:
            return ["DmlExecutionProvider"], [{}]
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
                if self.onnx_session is not None:
                    self.onnx_session.set_providers(
                        providers=providers,
                        provider_options=options,
                    )
                if self.content_vec_onnx is not None:
                    self.content_vec_onnx.set_providers(
                        providers=providers,
                        provider_options=options,
                    )

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

    def get_unit_f0(self, audio_buffer, tran):
        wav_44k = audio_buffer

        if self.settings.f0Detector == "dio":
            f0 = compute_f0_dio(
                wav_44k,
                sampling_rate=self.hps.data.sampling_rate,
                hop_length=self.hps.data.hop_length,
            )
        else:
            f0 = compute_f0_harvest(
                wav_44k,
                sampling_rate=self.hps.data.sampling_rate,
                hop_length=self.hps.data.hop_length,
            )

        if wav_44k.shape[0] % self.hps.data.hop_length != 0:
            print(
                f" !!! !!! !!! wav size not multiple of hopsize: {wav_44k.shape[0] / self.hps.data.hop_length}"
            )

        f0, uv = utils.interpolate_f0(f0)
        f0 = torch.FloatTensor(f0)
        uv = torch.FloatTensor(uv)
        f0 = f0 * 2 ** (tran / 12)
        f0 = f0.unsqueeze(0)
        uv = uv.unsqueeze(0)

        wav16k_numpy = librosa.resample(
            audio_buffer, orig_sr=self.hps.data.sampling_rate, target_sr=16000
        )
        wav16k_tensor = torch.from_numpy(wav16k_numpy)

        if (
            self.settings.gpu < 0 or self.gpu_num == 0
        ) or self.settings.framework == "ONNX":
            dev = torch.device("cpu")
        else:
            dev = torch.device("cuda", index=self.settings.gpu)

        if hasattr(self, "content_vec_onnx"):
            c = self.content_vec_onnx.run(
                ["units"],
                {
                    "audio": wav16k_numpy.reshape(1, -1),
                },
            )
            c = torch.from_numpy(np.array(c)).squeeze(0).transpose(1, 2)
            # print("onnx hubert:", self.content_vec_onnx.get_providers())
        else:
            if self.hps.model.ssl_dim == 768:
                self.hubert_model = self.hubert_model.to(dev)
                wav16k_tensor = wav16k_tensor.to(dev)
                c = get_hubert_content_layer9(
                    self.hubert_model, wav_16k_tensor=wav16k_tensor
                )
            else:
                self.hubert_model = self.hubert_model.to(dev)
                wav16k_tensor = wav16k_tensor.to(dev)
                c = utils.get_hubert_content(
                    self.hubert_model, wav_16k_tensor=wav16k_tensor
                )

        uv = uv.to(dev)
        f0 = f0.to(dev)

        c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[1])

        if (
            self.settings.clusterInferRatio != 0
            and hasattr(self, "cluster_model")
            and self.cluster_model is not None
        ):
            speaker = [
                key
                for key, value in self.settings.speakers.items()
                if value == self.settings.dstId
            ]
            if len(speaker) != 1:
                pass
                # print("not only one speaker found.", speaker)
            else:
                cluster_c = cluster.get_cluster_center_result(
                    self.cluster_model, c.cpu().numpy().T, speaker[0]
                ).T
                cluster_c = torch.FloatTensor(cluster_c).to(dev)
                c = c.to(dev)
                c = (
                    self.settings.clusterInferRatio * cluster_c
                    + (1 - self.settings.clusterInferRatio) * c
                )

        c = c.unsqueeze(0)
        return c, f0, uv

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

        convertSize = (
            inputSize + crossfadeSize + solaSearchFrame + self.settings.extraConvertSize
        )

        if convertSize % self.hps.data.hop_length != 0:  # モデルの出力のホップサイズで切り捨てが発生するので補う。
            convertSize = convertSize + (
                self.hps.data.hop_length - (convertSize % self.hps.data.hop_length)
            )

        convertOffset = -1 * convertSize
        self.audio_buffer = self.audio_buffer[convertOffset:]  # 変換対象の部分だけ抽出

        cropOffset = -1 * (inputSize + crossfadeSize)
        cropEnd = -1 * (crossfadeSize)
        crop = self.audio_buffer[cropOffset:cropEnd]

        rms = np.sqrt(np.square(crop).mean(axis=0))
        vol = max(rms, self.prevVol * 0.0)
        self.prevVol = vol

        c, f0, uv = self.get_unit_f0(self.audio_buffer, self.settings.tran)
        return (c, f0, uv, convertSize, vol)

    def _onnx_inference(self, data):
        if hasattr(self, "onnx_session") is False or self.onnx_session is None:
            print("[Voice Changer] No onnx session.")
            raise NoModeLoadedException("ONNX")

        convertSize = data[3]
        vol = data[4]
        data = (
            data[0],
            data[1],
            data[2],
        )

        if vol < self.settings.silentThreshold:
            return np.zeros(convertSize).astype(np.int16)

        c, f0, uv = [x.numpy() for x in data]
        sid_target = torch.LongTensor([self.settings.dstId]).unsqueeze(0).numpy()
        audio1 = (
            self.onnx_session.run(
                ["audio"],
                {
                    "c": c.astype(np.float32),
                    "f0": f0.astype(np.float32),
                    "uv": uv.astype(np.float32),
                    "g": sid_target.astype(np.int64),
                    "noise_scale": np.array([self.settings.noiseScale]).astype(
                        np.float32
                    ),
                    # "predict_f0": np.array([self.settings.dstId]).astype(np.int64),
                },
            )[0][0, 0]
            * self.hps.data.max_wav_value
        )

        audio1 = audio1 * vol
        result = audio1
        return result

    def _pyTorch_inference(self, data):
        if hasattr(self, "net_g") is False or self.net_g is None:
            print("[Voice Changer] No pyTorch session.")
            raise NoModeLoadedException("pytorch")

        if self.settings.gpu < 0 or self.gpu_num == 0:
            dev = torch.device("cpu")
        else:
            dev = torch.device("cuda", index=self.settings.gpu)

        convertSize = data[3]
        vol = data[4]
        data = (
            data[0],
            data[1],
            data[2],
        )

        if vol < self.settings.silentThreshold:
            return np.zeros(convertSize).astype(np.int16)

        with torch.no_grad():
            c, f0, uv = [x.to(dev) for x in data]
            sid_target = torch.LongTensor([self.settings.dstId]).to(dev).unsqueeze(0)
            self.net_g.to(dev)
            # audio1 = self.net_g.infer(c, f0=f0, g=sid_target, uv=uv, predict_f0=True, noice_scale=0.1)[0][0, 0].data.float()
            predict_f0_flag = True if self.settings.predictF0 == 1 else False
            audio1 = self.net_g.infer(
                c,
                f0=f0,
                g=sid_target,
                uv=uv,
                predict_f0=predict_f0_flag,
                noice_scale=self.settings.noiseScale,
            )
            audio1 = audio1[0][0].data.float()
            # audio1 = self.net_g.infer(c, f0=f0, g=sid_target, uv=uv, predict_f0=predict_f0_flag,
            #                           noice_scale=self.settings.noiceScale)[0][0, 0].data.float()
            audio1 = audio1 * self.hps.data.max_wav_value

            audio1 = audio1 * vol

            result = audio1.float().cpu().numpy()

            # result = infer_tool.pad_array(result, length)
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
        remove_path = os.path.join("so-vits-svc-40")
        sys.path = [x for x in sys.path if x.endswith(remove_path) is False]

        for key in list(sys.modules):
            val = sys.modules.get(key)
            try:
                file_path = val.__file__
                if file_path.find("so-vits-svc-40" + os.path.sep) >= 0:
                    # print("remove", key, file_path)
                    sys.modules.pop(key)
            except Exception:  # type:ignore
                pass


def resize_f0(x, target_len):
    source = np.array(x)
    source[source < 0.001] = np.nan
    target = np.interp(
        np.arange(0, len(source) * target_len, len(source)) / target_len,
        np.arange(0, len(source)),
        source,
    )
    res = np.nan_to_num(target)
    return res


def compute_f0_dio(wav_numpy, p_len=None, sampling_rate=44100, hop_length=512):
    if p_len is None:
        p_len = wav_numpy.shape[0] // hop_length
    f0, t = pw.dio(
        wav_numpy.astype(np.double),
        fs=sampling_rate,
        f0_ceil=800,
        frame_period=1000 * hop_length / sampling_rate,
    )
    f0 = pw.stonemask(wav_numpy.astype(np.double), f0, t, sampling_rate)
    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)
    return resize_f0(f0, p_len)


def compute_f0_harvest(wav_numpy, p_len=None, sampling_rate=44100, hop_length=512):
    if p_len is None:
        p_len = wav_numpy.shape[0] // hop_length
    f0, t = pw.harvest(
        wav_numpy.astype(np.double),
        fs=sampling_rate,
        frame_period=5.5,
        f0_floor=71.0,
        f0_ceil=1000.0,
    )

    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)
    return resize_f0(f0, p_len)


def get_hubert_content_layer9(hmodel, wav_16k_tensor):
    feats = wav_16k_tensor
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    feats = feats.view(1, -1)
    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
    inputs = {
        "source": feats.to(wav_16k_tensor.device),
        "padding_mask": padding_mask.to(wav_16k_tensor.device),
        "output_layer": 9,  # layer 9
    }
    with torch.no_grad():
        logits = hmodel.extract_features(**inputs)

    return logits[0].transpose(1, 2)
