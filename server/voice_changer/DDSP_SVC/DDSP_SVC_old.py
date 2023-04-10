import sys
import os
if sys.platform.startswith('darwin'):
    baseDir = [x for x in sys.path if x.endswith("Contents/MacOS")]
    if len(baseDir) != 1:
        print("baseDir should be only one ", baseDir)
        sys.exit()
    modulePath = os.path.join(baseDir[0], "DDSP-SVC")
    sys.path.append(modulePath)
else:
    sys.path.append("DDSP-SVC")

import io
from dataclasses import dataclass, asdict, field
from functools import reduce
import numpy as np
import torch
import onnxruntime
import pyworld as pw
import ddsp.vocoder as vo

import librosa
providers = ['OpenVINOExecutionProvider', "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]


@dataclass
class DDSP_SVCSettings():
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


class DDSP_SVC:
    def __init__(self, params):
        self.settings = DDSP_SVCSettings()
        self.net_g = None
        self.onnx_session = None

        self.raw_path = io.BytesIO()
        self.gpu_num = torch.cuda.device_count()
        self.prevVol = 0
        self.params = params
        print("DDSP-SVC initialization:", params)

    def loadModel(self, config: str, pyTorch_model_file: str = None, onnx_model_file: str = None, clusterTorchModel: str = None):

        self.settings.configFile = config
        # model
        model, args = vo.load_model(pyTorch_model_file)

        # hubert
        self.model = model
        self.args = args

        vec_path = self.params["hubert"]
        self.encoder = vo.Units_Encoder(
            args.data.encoder,
            vec_path,
            args.data.encoder_sample_rate,
            args.data.encoder_hop_size,
            device="cpu")
        # f0dec
        self.f0_detector = vo.F0_Extractor(
            self.settings.f0Detector,
            44100,
            512,
            float(50),
            float(1100))

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
        return 44100

    def get_unit_f0(self, audio_buffer, tran):
        if (self.settings.gpu < 0 or self.gpu_num == 0) or self.settings.framework == "ONNX":
            dev = torch.device("cpu")
        else:
            dev = torch.device("cpu")
            # dev = torch.device("cuda", index=self.settings.gpu)

        wav_44k = audio_buffer
        f0 = self.f0_detector.extract(wav_44k, uv_interp=True, device=dev)
        f0 = torch.from_numpy(f0).float().to(dev).unsqueeze(-1).unsqueeze(0)
        f0 = f0 * 2 ** (float(10) / 12)
        # print("f0:", f0)

        print("wav_44k:::", wav_44k)
        c = self.encoder.encode(torch.from_numpy(audio_buffer).float().unsqueeze(0).to(dev), 44100, 512)
        # print("c:", c)
        return c, f0

    def generate_input(self, newData: any, inputSize: int, crossfadeSize: int):
        # newData = newData.astype(np.float32) / 32768.0
        # newData = newData.astype(np.float32) / self.hps.data.max_wav_value

        if hasattr(self, "audio_buffer"):
            self.audio_buffer = np.concatenate([self.audio_buffer, newData], 0)  # 過去のデータに連結
        else:
            self.audio_buffer = newData

        convertSize = inputSize + crossfadeSize + self.settings.extraConvertSize
        hop_size = int(self.args.data.block_size * 44100 / self.args.data.sampling_rate)
        print("hopsize", hop_size)
        if convertSize % hop_size != 0:  # モデルの出力のホップサイズで切り捨てが発生するので補う。
            convertSize = convertSize + (hop_size - (convertSize % hop_size))

        print("convsize", convertSize)
        self.audio_buffer = self.audio_buffer[-1 * convertSize:]  # 変換対象の部分だけ抽出

        crop = self.audio_buffer[-1 * (inputSize + crossfadeSize):-1 * (crossfadeSize)]

        rms = np.sqrt(np.square(crop).mean(axis=0))
        vol = max(rms, self.prevVol * 0.0)
        self.prevVol = vol

        c, f0 = self.get_unit_f0(self.audio_buffer, self.settings.tran)
        return (c, f0, convertSize, vol)

    def _onnx_inference(self, data):
        if hasattr(self, "onnx_session") == False or self.onnx_session == None:
            print("[Voice Changer] No onnx session.")
            return np.zeros(1).astype(np.int16)

        c = data[0]
        f0 = data[1]
        convertSize = data[2]
        vol = data[3]

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

        if hasattr(self, "model") == False or self.model == None:
            print("[Voice Changer] No pyTorch session.")
            return np.zeros(1).astype(np.int16)

        if self.settings.gpu < 0 or self.gpu_num == 0:
            dev = torch.device("cpu")
        else:
            dev = torch.device("cpu")
            # dev = torch.device("cuda", index=self.settings.gpu)

        c = data[0]
        f0 = data[1]
        convertSize = data[2]
        vol = data[3]
        if vol < self.settings.silentThreshold:
            return np.zeros(convertSize).astype(np.int16)

        with torch.no_grad():
            c.to(dev)
            f0.to(dev)
            vol = torch.from_numpy(np.array([vol] * c.shape[1])).float().to(dev).unsqueeze(-1).unsqueeze(0)
            spk_id = torch.LongTensor(np.array([[1]])).to(dev)
            # print("vol", vol)
            print("input", c.shape, f0.shape)
            seg_output, _, (s_h, s_n) = self.model(c, f0, vol, spk_id=spk_id)

            seg_output = seg_output.squeeze().cpu().numpy()
            print("SEG:", seg_output)

        return seg_output

    def inference(self, data):
        if self.settings.framework == "ONNX":
            audio = self._onnx_inference(data)
        else:
            audio = self._pyTorch_inference(data)
        return audio

    def destroy(self):
        del self.net_g
        del self.onnx_session
