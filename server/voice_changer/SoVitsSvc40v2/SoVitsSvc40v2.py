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
from dataclasses import dataclass, asdict
from functools import reduce
import numpy as np
import torch
import onnxruntime
import pyworld as pw

from models import SynthesizerTrn
import utils
from fairseq import checkpoint_utils
import librosa
from inference import infer_tool
providers = ['OpenVINOExecutionProvider', "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]


@dataclass
class SoVitsSvc40v2Settings():
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


class SoVitsSvc40v2:
    def __init__(self):
        self.settings = SoVitsSvc40v2Settings()
        self.net_g = None
        self.onnx_session = None

        self.raw_path = io.BytesIO()
        self.gpu_num = torch.cuda.device_count()
        self.prevVol = 0

    def loadModel(self, config: str, pyTorch_model_file: str = None, onnx_model_file: str = None):
        self.settings.configFile = config
        self.hps = utils.get_hparams_from_file(config)

        # hubert model
        print("loading hubert model")
        vec_path = "hubert/checkpoint_best_legacy_500.pt"
        print("load model(s) from {}".format(vec_path))
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [vec_path],
            suffix="",
        )
        model = models[0]
        model.eval()
        self.hubert_model = utils.get_hubert_model().cpu()

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

        # # ONNXモデル生成
        # if onnx_model_file != None:
        #     ort_options = onnxruntime.SessionOptions()
        #     ort_options.intra_op_num_threads = 8
        #     self.onnx_session = onnxruntime.InferenceSession(
        #         onnx_model_file,
        #         providers=providers
        #     )
        return self.get_info()

    def update_setteings(self, key: str, val: any):
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
        f0 = utils.compute_f0_dio(wav_44k, sampling_rate=self.hps.data.sampling_rate, hop_length=self.hps.data.hop_length)
        print(f"--- >>>>> ---- >>>> {wav_44k.shape[0] / self.hps.data.hop_length}")

        f0, uv = utils.interpolate_f0(f0)
        f0 = torch.FloatTensor(f0)
        uv = torch.FloatTensor(uv)
        f0 = f0 * 2 ** (tran / 12)
        f0 = f0.unsqueeze(0)
        uv = uv.unsqueeze(0)

        # wav16k = librosa.resample(audio_buffer, orig_sr=24000, target_sr=16000)
        wav16k = librosa.resample(audio_buffer, orig_sr=self.hps.data.sampling_rate, target_sr=16000)
        wav16k = torch.from_numpy(wav16k)
        c = utils.get_hubert_content(self.hubert_model, wav_16k_tensor=wav16k)
        c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[1])
        c = c.unsqueeze(0)
        return c, f0, uv

    def generate_input(self, newData: any, convertSize: int, cropRange):
        newData = newData.astype(np.float32) / self.hps.data.max_wav_value

        if hasattr(self, "audio_buffer"):
            self.audio_buffer = np.concatenate([self.audio_buffer, newData], 0)  # 過去のデータに連結
        else:
            self.audio_buffer = newData

        self.audio_buffer = self.audio_buffer[-(convertSize):]  # 変換対象の部分だけ抽出

        crop = self.audio_buffer[cropRange[0]:cropRange[1]]

        rms = np.sqrt(np.square(crop).mean(axis=0))
        vol = max(rms, self.prevVol * 0.1)
        self.prevVol = vol
        # print(f"         Crop:{crop.shape}, vol{vol}")

        c, f0, uv = self.get_unit_f0(self.audio_buffer, 20)
        return (c, f0, uv, convertSize, vol)

    def _onnx_inference(self, data):
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

        # if vol < 0.00001:
        #     print("silcent")
        #     return np.zeros(convertSize).astype(np.int16)
        # print(vol)

        with torch.no_grad():
            c, f0, uv = [x.to(dev)for x in data]
            sid_target = torch.LongTensor([0]).to(dev)
            self.net_g.to(dev)
            # audio1 = self.net_g.infer(c, f0=f0, g=sid_target, uv=uv, predict_f0=True, noice_scale=0.1)[0][0, 0].data.float()
            audio1 = self.net_g.infer(c, f0=f0, g=sid_target, uv=uv, predict_f0=False, noice_scale=0.4)[0][0, 0].data.float()
            audio1 = audio1 * self.hps.data.max_wav_value

            result = audio1.float().cpu().numpy()

            # result = infer_tool.pad_array(result, length)
        return result

    def inference(self, data):
        if self.settings.framework == "ONNX":
            audio = self._onnx_inference(data)
        else:
            audio = self._pyTorch_inference(data)
        return audio

    def destroy(self):
        del self.net_g
        del self.onnx_session
