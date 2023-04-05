import sys
import os

sys.argv = ["MMVCServerSIO.py"]

if sys.platform.startswith('darwin'):
    baseDir = [x for x in sys.path if x.endswith("Contents/MacOS")]
    if len(baseDir) != 1:
        print("baseDir should be only one ", baseDir)
        sys.exit()
    modulePath = os.path.join(baseDir[0], "RVC")
    sys.path.append(modulePath)
else:
    sys.path.append("RVC")

print("RVC 3")
import io
from dataclasses import dataclass, asdict, field
from functools import reduce
import numpy as np
import torch
import onnxruntime
# onnxruntime.set_default_logger_severity(3)
from const import HUBERT_ONNX_MODEL_PATH

import pyworld as pw

from vc_infer_pipeline import VC
from infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from fairseq import checkpoint_utils
providers = ['OpenVINOExecutionProvider', "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]


@dataclass
class RVCSettings():
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


class RVC:
    def __init__(self, params):
        self.settings = RVCSettings()
        self.net_g = None
        self.onnx_session = None

        self.raw_path = io.BytesIO()
        self.gpu_num = torch.cuda.device_count()
        self.prevVol = 0
        self.params = params
        print("RVC initialization: ", params)

    def loadModel(self, config: str, pyTorch_model_file: str = None, onnx_model_file: str = None, clusterTorchModel: str = None):
        self.device = torch.device("cuda", index=self.settings.gpu)
        self.settings.configFile = config
        # self.hps = utils.get_hparams_from_file(config)
        # self.settings.speakers = self.hps.spk

        # hubert model
        try:
            # hubert_path = self.params["hubert"]
            # useHubertOnnx = self.params["useHubertOnnx"]
            # self.useHubertOnnx = useHubertOnnx

            # if useHubertOnnx == True:
            #     ort_options = onnxruntime.SessionOptions()
            #     ort_options.intra_op_num_threads = 8
            #     self.hubert_onnx = onnxruntime.InferenceSession(
            #         HUBERT_ONNX_MODEL_PATH,
            #         providers=providers
            #     )
            # else:
            #     models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            #         [hubert_path],
            #         suffix="",
            #     )
            #     model = models[0]
            #     model.eval()
            #     self.hubert_model = model.cpu()
            models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(["hubert_base.pt"], suffix="",)
            model = models[0]
            model.eval()
            # model = model.half()
            self.hubert_model = model
            self.hubert_model = self.hubert_model.to(self.device)

        except Exception as e:
            print("EXCEPTION during loading hubert/contentvec model", e)

        if pyTorch_model_file != None:
            self.settings.pyTorchModelFile = pyTorch_model_file
        if onnx_model_file:
            self.settings.onnxModelFile = onnx_model_file

        # PyTorchモデル生成
        if pyTorch_model_file != None:
            cpt = torch.load(pyTorch_model_file, map_location="cpu")
            self.tgt_sr = cpt["config"][-1]
            # n_spk = cpt["config"][-3]
            is_half = False
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
            net_g.eval()
            net_g.load_state_dict(cpt["weight"], strict=False)
            # net_g = net_g.half()
            self.net_g = net_g
            self.net_g = self.net_g.to(self.device)

            # self.net_g = SynthesizerTrn(
            #     self.hps.data.filter_length // 2 + 1,
            #     self.hps.train.segment_size // self.hps.data.hop_length,
            #     **self.hps.model
            # )
            # self.net_g.eval()
            # utils.load_checkpoint(pyTorch_model_file, self.net_g, None)

        # ONNXモデル生成
        if onnx_model_file != None:
            ort_options = onnxruntime.SessionOptions()
            ort_options.intra_op_num_threads = 8
            self.onnx_session = onnxruntime.InferenceSession(
                onnx_model_file,
                providers=providers
            )
            # input_info = self.onnx_session.get_inputs()
            # for i in input_info:
            #     print("input", i)
            # output_info = self.onnx_session.get_outputs()
            # for i in output_info:
            #     print("output", i)
        return self.get_info()

    def update_setteings(self, key: str, val: any):
        if key == "onnxExecutionProvider" and self.onnx_session != None:
            if val == "CUDAExecutionProvider":
                if self.settings.gpu < 0 or self.settings.gpu >= self.gpu_num:
                    self.settings.gpu = 0
                provider_options = [{'device_id': self.settings.gpu}]
                self.onnx_session.set_providers(providers=[val], provider_options=provider_options)
                if hasattr(self, "hubert_onnx"):
                    self.hubert_onnx.set_providers(providers=[val], provider_options=provider_options)
            else:
                self.onnx_session.set_providers(providers=[val])
                if hasattr(self, "hubert_onnx"):
                    self.hubert_onnx.set_providers(providers=[val])
        elif key == "onnxExecutionProvider" and self.onnx_session == None:
            print("Onnx is not enabled. Please load model.")
            return False
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
        return self.tgt_sr
        # return 24000

    def generate_input(self, newData: any, inputSize: int, crossfadeSize: int):
        # import wave
        # filename = "testc2.wav"
        # if os.path.exists(filename):
        #     print("[IORecorder] delete old analyze file.", filename)
        #     os.remove(filename)
        # fo = wave.open(filename, 'wb')
        # fo.setnchannels(1)
        # fo.setsampwidth(2)
        # # fo.setframerate(24000)
        # fo.setframerate(self.tgt_sr)
        # fo.writeframes(newData.astype(np.int16))
        # fo.close()

        # newData = newData.astype(np.float32) / self.hps.data.max_wav_value
        newData = newData.astype(np.float32) / 32768.0

        if hasattr(self, "audio_buffer"):
            self.audio_buffer = np.concatenate([self.audio_buffer, newData], 0)  # 過去のデータに連結
        else:
            self.audio_buffer = newData

        convertSize = inputSize + crossfadeSize + self.settings.extraConvertSize

        # if convertSize % self.hps.data.hop_length != 0:  # モデルの出力のホップサイズで切り捨てが発生するので補う。
        if convertSize % 128 != 0:  # モデルの出力のホップサイズで切り捨てが発生するので補う。
            # convertSize = convertSize + (self.hps.data.hop_length - (convertSize % self.hps.data.hop_length))
            convertSize = convertSize + (128 - (convertSize % 128))

        self.audio_buffer = self.audio_buffer[-1 * convertSize:]  # 変換対象の部分だけ抽出

        crop = self.audio_buffer[-1 * (inputSize + crossfadeSize):-1 * (crossfadeSize)]
        rms = np.sqrt(np.square(crop).mean(axis=0))
        vol = max(rms, self.prevVol * 0.0)
        self.prevVol = vol

        import wave
        filename = "testc2.wav"
        if os.path.exists(filename):
            print("[IORecorder] delete old analyze file.", filename)
            os.remove(filename)
        fo = wave.open(filename, 'wb')
        fo.setnchannels(1)
        fo.setsampwidth(2)
        # fo.setframerate(24000)
        fo.setframerate(self.tgt_sr)
        fo.writeframes((self.audio_buffer * 32768.0).astype(np.int16))
        fo.close()

        return (self.audio_buffer, convertSize, vol)

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

        audio = data[0]
        convertSize = data[1]
        vol = data[2]
        # from scipy.io import wavfile
        # # wavfile.write("testa.wav", self.tgt_sr, audio * 32768.0)
        # wavfile.write("testa.wav", 24000, audio * 32768.0)

        filename = "testc2.wav"
        audio = load_audio(filename, 16000)

        if vol < self.settings.silentThreshold:
            return np.zeros(convertSize).astype(np.int16)

        is_half = False
        with torch.no_grad():
            vc = VC(self.tgt_sr, dev, is_half)
            sid = 0
            times = [0, 0, 0]
            f0_up_key = 0
            f0_method = "pm"
            file_index = ""
            file_big_npy = ""
            index_rate = 1
            if_f0 = 1
            f0_file = None

            audio_out = vc.pipeline(self.hubert_model, self.net_g, sid, audio, times, f0_up_key, f0_method,
                                    file_index, file_big_npy, index_rate, if_f0, f0_file=f0_file)
            result = audio_out
        from scipy.io import wavfile
        wavfile.write("testaaaaa.wav", self.tgt_sr, result)
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


# def resize_f0(x, target_len):
#     source = np.array(x)
#     source[source < 0.001] = np.nan
#     target = np.interp(np.arange(0, len(source) * target_len, len(source)) / target_len, np.arange(0, len(source)), source)
#     res = np.nan_to_num(target)
#     return res


# def compute_f0_dio(wav_numpy, p_len=None, sampling_rate=44100, hop_length=512):
#     if p_len is None:
#         p_len = wav_numpy.shape[0] // hop_length
#     f0, t = pw.dio(
#         wav_numpy.astype(np.double),
#         fs=sampling_rate,
#         f0_ceil=800,
#         frame_period=1000 * hop_length / sampling_rate,
#     )
#     f0 = pw.stonemask(wav_numpy.astype(np.double), f0, t, sampling_rate)
#     for index, pitch in enumerate(f0):
#         f0[index] = round(pitch, 1)
#     return resize_f0(f0, p_len)


# def compute_f0_harvest(wav_numpy, p_len=None, sampling_rate=44100, hop_length=512):
#     if p_len is None:
#         p_len = wav_numpy.shape[0] // hop_length
#     f0, t = pw.harvest(wav_numpy.astype(np.double), fs=sampling_rate, frame_period=5.5, f0_floor=71.0, f0_ceil=1000.0)

#     for index, pitch in enumerate(f0):
#         f0[index] = round(pitch, 1)
#     return resize_f0(f0, p_len)


# def get_hubert_content_layer9(hmodel, wav_16k_tensor):
#     feats = wav_16k_tensor
#     if feats.dim() == 2:  # double channels
#         feats = feats.mean(-1)
#     assert feats.dim() == 1, feats.dim()
#     feats = feats.view(1, -1)
#     padding_mask = torch.BoolTensor(feats.shape).fill_(False)
#     inputs = {
#         "source": feats.to(wav_16k_tensor.device),
#         "padding_mask": padding_mask.to(wav_16k_tensor.device),
#         "output_layer": 9,  # layer 9
#     }
#     with torch.no_grad():
#         logits = hmodel.extract_features(**inputs)

#     return logits[0].transpose(1, 2)


import ffmpeg


def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
