import sys
import os
if sys.platform.startswith('darwin'):
    baseDir = [x for x in sys.path if x.endswith("Contents/MacOS")]
    if len(baseDir) != 1:
        print("baseDir should be only one ", baseDir)
        sys.exit()
    modulePath = os.path.join(baseDir[0], "so-vits-svc-40v2")
    sys.path.append(modulePath)
else:
    sys.path.append("so-vits-svc-40v2")

import io
from dataclasses import dataclass, asdict, field
from functools import reduce
import numpy as np
import torch
import onnxruntime
import pyworld as pw

from models import SynthesizerTrn
import cluster
import utils
from fairseq import checkpoint_utils
import librosa
providers = ['OpenVINOExecutionProvider', "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]


@dataclass
class SoVitsSvc40v2Settings():
    gpu: int = 0
    dstId: int = 0

    f0Detector: str = "dio"  # dio or harvest
    tran: int = 20
    noiceScale: float = 0.3
    predictF0: int = 0  # 0:False, 1:True
    silentThreshold: float = 0.00001
    extraConvertSize: int = 1024 * 32
    clusterInferRatio: float = 0.1

    framework: str = "PyTorch"  # PyTorch or ONNX
    pyTorchModelFile: str = ""
    onnxModelFile: str = ""
    configFile: str = ""

    speakers: dict[str, int] = field(
        default_factory=lambda: {}
    )

    # ↓mutableな物だけ列挙
    intData = ["gpu", "dstId", "tran", "predictF0", "extraConvertSize"]
    floatData = ["noiceScale", "silentThreshold", "clusterInferRatio"]
    strData = ["framework", "f0Detector"]


class SoVitsSvc40v2:
    def __init__(self, params):
        self.settings = SoVitsSvc40v2Settings()
        self.net_g = None
        self.onnx_session = None

        self.raw_path = io.BytesIO()
        self.gpu_num = torch.cuda.device_count()
        self.prevVol = 0
        self.params = params
        print("so-vits-svc 40v2 initialization:", params)

    def loadModel(self, config: str, pyTorch_model_file: str = None, onnx_model_file: str = None, clusterTorchModel: str = None):

        self.settings.configFile = config
        self.hps = utils.get_hparams_from_file(config)
        self.settings.speakers = self.hps.spk

        # hubert model
        try:
            # if sys.platform.startswith('darwin'):
            #     vec_path = os.path.join(sys._MEIPASS, "hubert/checkpoint_best_legacy_500.pt")
            # else:
            #     vec_path = "hubert/checkpoint_best_legacy_500.pt"
            vec_path = self.params["hubert"]

            models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
                [vec_path],
                suffix="",
            )
            model = models[0]
            model.eval()
            self.hubert_model = model.cpu()
        except Exception as e:
            print("EXCEPTION during loading hubert/contentvec model", e)

        # cluster
        try:
            if clusterTorchModel != None and os.path.exists(clusterTorchModel):
                self.cluster_model = cluster.get_cluster_model(clusterTorchModel)
            else:
                self.cluster_model = None
        except Exception as e:
            print("EXCEPTION during loading cluster model ", e)

        if pyTorch_model_file != None:
            self.settings.pyTorchModelFile = pyTorch_model_file
        if onnx_model_file:
            self.settings.onnxModelFile = onnx_model_file

        # PyTorchモデル生成
        if pyTorch_model_file != None:
            self.net_g = SynthesizerTrn(
                self.hps
            )
            self.net_g.eval()
            utils.load_checkpoint(pyTorch_model_file, self.net_g, None)

        # ONNXモデル生成
        if onnx_model_file != None:
            ort_options = onnxruntime.SessionOptions()
            ort_options.intra_op_num_threads = 8
            self.onnx_session = onnxruntime.InferenceSession(
                onnx_model_file,
                providers=providers
            )
            input_info = self.onnx_session.get_inputs()
        return self.get_info()

    def update_settings(self, key: str, val: any):
        if key == "onnxExecutionProvider" and self.onnx_session != None:
            if val == "CUDAExecutionProvider":
                if self.settings.gpu < 0 or self.settings.gpu >= self.gpu_num:
                    self.settings.gpu = 0
                provider_options = [{'device_id': self.settings.gpu}]
                self.onnx_session.set_providers(providers=[val], provider_options=provider_options)
            else:
                self.onnx_session.set_providers(providers=[val])
        elif key in self.settings.intData:
            setattr(self.settings, key, int(val))
            if key == "gpu" and val >= 0 and val < self.gpu_num and self.onnx_session != None:
                providers = self.onnx_session.get_providers()
                print("Providers:", providers)
                if "CUDAExecutionProvider" in providers:
                    provider_options = [{'device_id': self.settings.gpu}]
                    self.onnx_session.set_providers(providers=["CUDAExecutionProvider"], provider_options=provider_options)
        elif key in self.settings.floatData:
            setattr(self.settings, key, float(val))
        elif key in self.settings.strData:
            setattr(self.settings, key, str(val))
        else:
            return False

        return True

    def get_info(self):
        data = asdict(self.settings)

        data["onnxExecutionProviders"] = self.onnx_session.get_providers() if self.onnx_session != None else []
        files = ["configFile", "pyTorchModelFile", "onnxModelFile"]
        for f in files:
            if data[f] != None and os.path.exists(data[f]):
                data[f] = os.path.basename(data[f])
            else:
                data[f] = ""

        return data

    def get_processing_sampling_rate(self):
        return self.hps.data.sampling_rate

    def get_unit_f0(self, audio_buffer, tran):
        wav_44k = audio_buffer
        # f0 = utils.compute_f0_parselmouth(wav, sampling_rate=self.target_sample, hop_length=self.hop_size)
        # f0 = utils.compute_f0_dio(wav_44k, sampling_rate=self.hps.data.sampling_rate, hop_length=self.hps.data.hop_length)
        if self.settings.f0Detector == "dio":
            f0 = compute_f0_dio(wav_44k, sampling_rate=self.hps.data.sampling_rate, hop_length=self.hps.data.hop_length)
        else:
            f0 = compute_f0_harvest(wav_44k, sampling_rate=self.hps.data.sampling_rate, hop_length=self.hps.data.hop_length)

        if wav_44k.shape[0] % self.hps.data.hop_length != 0:
            print(f" !!! !!! !!! wav size not multiple of hopsize: {wav_44k.shape[0] / self.hps.data.hop_length}")

        f0, uv = utils.interpolate_f0(f0)
        f0 = torch.FloatTensor(f0)
        uv = torch.FloatTensor(uv)
        f0 = f0 * 2 ** (tran / 12)
        f0 = f0.unsqueeze(0)
        uv = uv.unsqueeze(0)

        # wav16k = librosa.resample(audio_buffer, orig_sr=24000, target_sr=16000)
        wav16k = librosa.resample(audio_buffer, orig_sr=self.hps.data.sampling_rate, target_sr=16000)
        wav16k = torch.from_numpy(wav16k)

        if (self.settings.gpu < 0 or self.gpu_num == 0) or self.settings.framework == "ONNX":
            dev = torch.device("cpu")
        else:
            dev = torch.device("cuda", index=self.settings.gpu)

        self.hubert_model = self.hubert_model.to(dev)
        wav16k = wav16k.to(dev)
        uv = uv.to(dev)
        f0 = f0.to(dev)

        c = utils.get_hubert_content(self.hubert_model, wav_16k_tensor=wav16k)
        c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[1])

        if self.settings.clusterInferRatio != 0 and hasattr(self, "cluster_model") and self.cluster_model != None:
            speaker = [key for key, value in self.settings.speakers.items() if value == self.settings.dstId]
            if len(speaker) != 1:
                print("not only one speaker found.", speaker)
            else:
                cluster_c = cluster.get_cluster_center_result(self.cluster_model, c.cpu().numpy().T, speaker[0]).T
                # cluster_c = cluster.get_cluster_center_result(self.cluster_model, c.cpu().numpy().T, self.settings.dstId).T
                cluster_c = torch.FloatTensor(cluster_c).to(dev)
                # print("cluster DEVICE", cluster_c.device, c.device)
                c = self.settings.clusterInferRatio * cluster_c + (1 - self.settings.clusterInferRatio) * c

        c = c.unsqueeze(0)
        return c, f0, uv

    def generate_input(self, newData: any, inputSize: int, crossfadeSize: int):
        newData = newData.astype(np.float32) / self.hps.data.max_wav_value

        if hasattr(self, "audio_buffer"):
            self.audio_buffer = np.concatenate([self.audio_buffer, newData], 0)  # 過去のデータに連結
        else:
            self.audio_buffer = newData

        convertSize = inputSize + crossfadeSize + self.settings.extraConvertSize

        if convertSize % self.hps.data.hop_length != 0:  # モデルの出力のホップサイズで切り捨てが発生するので補う。
            convertSize = convertSize + (self.hps.data.hop_length - (convertSize % self.hps.data.hop_length))

        self.audio_buffer = self.audio_buffer[-1 * convertSize:]  # 変換対象の部分だけ抽出

        crop = self.audio_buffer[-1 * (inputSize + crossfadeSize):-1 * (crossfadeSize)]

        rms = np.sqrt(np.square(crop).mean(axis=0))
        vol = max(rms, self.prevVol * 0.0)
        self.prevVol = vol

        c, f0, uv = self.get_unit_f0(self.audio_buffer, self.settings.tran)
        return (c, f0, uv, convertSize, vol)

    def _onnx_inference(self, data):
        if hasattr(self, "onnx_session") == False or self.onnx_session == None:
            print("[Voice Changer] No onnx session.")
            return np.zeros(1).astype(np.int16)

        convertSize = data[3]
        vol = data[4]
        data = (data[0], data[1], data[2],)

        if vol < self.settings.silentThreshold:
            return np.zeros(convertSize).astype(np.int16)

        c, f0, uv = [x.numpy() for x in data]
        audio1 = self.onnx_session.run(
            ["audio"],
            {
                "c": c,
                "f0": f0,
                "g": np.array([self.settings.dstId]).astype(np.int64),
                "uv": np.array([self.settings.dstId]).astype(np.int64),
                "predict_f0": np.array([self.settings.dstId]).astype(np.int64),
                "noice_scale": np.array([self.settings.dstId]).astype(np.int64),


            })[0][0, 0] * self.hps.data.max_wav_value

        audio1 = audio1 * vol

        result = audio1

        return result

        pass

    def _pyTorch_inference(self, data):
        if hasattr(self, "net_g") == False or self.net_g == None:
            print("[Voice Changer] No pyTorch session.")
            return np.zeros(1).astype(np.int16)

        if self.settings.gpu < 0 or self.gpu_num == 0:
            dev = torch.device("cpu")
        else:
            dev = torch.device("cuda", index=self.settings.gpu)

        convertSize = data[3]
        vol = data[4]
        data = (data[0], data[1], data[2],)

        if vol < self.settings.silentThreshold:
            return np.zeros(convertSize).astype(np.int16)

        with torch.no_grad():
            c, f0, uv = [x.to(dev)for x in data]
            sid_target = torch.LongTensor([self.settings.dstId]).to(dev)
            self.net_g.to(dev)
            # audio1 = self.net_g.infer(c, f0=f0, g=sid_target, uv=uv, predict_f0=True, noice_scale=0.1)[0][0, 0].data.float()
            predict_f0_flag = True if self.settings.predictF0 == 1 else False
            audio1 = self.net_g.infer(c, f0=f0, g=sid_target, uv=uv, predict_f0=predict_f0_flag,
                                      noice_scale=self.settings.noiceScale)[0][0, 0].data.float()
            audio1 = audio1 * self.hps.data.max_wav_value

            audio1 = audio1 * vol

            result = audio1.float().cpu().numpy()

            # result = infer_tool.pad_array(result, length)
        return result

    def inference(self, data):
        if self.settings.framework == "ONNX":
            audio = self._onnx_inference(data)
        else:
            audio = self._pyTorch_inference(data)
        return audio

    def __del__(self):
        del self.net_g
        del self.onnx_session

        remove_path = os.path.join("so-vits-svc-40v2")
        sys.path = [x for x in sys.path if x.endswith(remove_path) == False]

        for key in list(sys.modules):
            val = sys.modules.get(key)
            try:
                file_path = val.__file__
                if file_path.find("so-vits-svc-40v2" + os.path.sep) >= 0:
                    print("remove", key, file_path)
                    sys.modules.pop(key)
            except Exception as e:
                pass


def resize_f0(x, target_len):
    source = np.array(x)
    source[source < 0.001] = np.nan
    target = np.interp(np.arange(0, len(source) * target_len, len(source)) / target_len, np.arange(0, len(source)), source)
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
    f0, t = pw.harvest(wav_numpy.astype(np.double), fs=sampling_rate, frame_period=5.5, f0_floor=71.0, f0_ceil=1000.0)

    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)
    return resize_f0(f0, p_len)
