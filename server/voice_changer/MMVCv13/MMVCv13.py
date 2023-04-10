import sys
import os
if sys.platform.startswith('darwin'):
    baseDir = [x for x in sys.path if x.endswith("Contents/MacOS")]
    if len(baseDir) != 1:
        print("baseDir should be only one ", baseDir)
        sys.exit()
    modulePath = os.path.join(baseDir[0], "MMVC_Client_v13", "python")
    sys.path.append(modulePath)
else:
    sys.path.append("MMVC_Client_v13/python")


from dataclasses import dataclass, asdict
import numpy as np
import torch
import onnxruntime
import pyworld as pw

from symbols import symbols
from models import SynthesizerTrn
from voice_changer.MMVCv13.TrainerFunctions import TextAudioSpeakerCollate, spectrogram_torch, load_checkpoint, get_hparams_from_file

providers = ['OpenVINOExecutionProvider', "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]


@dataclass
class MMVCv13Settings():
    gpu: int = 0
    srcId: int = 0
    dstId: int = 101

    framework: str = "PyTorch"  # PyTorch or ONNX
    pyTorchModelFile: str = ""
    onnxModelFile: str = ""
    configFile: str = ""

    # ↓mutableな物だけ列挙
    intData = ["gpu", "srcId", "dstId"]
    floatData = []
    strData = ["framework"]


class MMVCv13:
    def __init__(self):
        self.settings = MMVCv13Settings()
        self.net_g = None
        self.onnx_session = None

        self.gpu_num = torch.cuda.device_count()
        self.text_norm = torch.LongTensor([0, 6, 0])

    def loadModel(self, config: str, pyTorch_model_file: str = None, onnx_model_file: str = None):
        self.settings.configFile = config
        self.hps = get_hparams_from_file(config)

        if pyTorch_model_file != None:
            self.settings.pyTorchModelFile = pyTorch_model_file
        if onnx_model_file:
            self.settings.onnxModelFile = onnx_model_file

        # PyTorchモデル生成
        if pyTorch_model_file != None:
            self.net_g = SynthesizerTrn(
                len(symbols),
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers=self.hps.data.n_speakers,
                **self.hps.model)
            self.net_g.eval()
            load_checkpoint(pyTorch_model_file, self.net_g, None)

        # ONNXモデル生成
        if onnx_model_file != None:
            ort_options = onnxruntime.SessionOptions()
            ort_options.intra_op_num_threads = 8
            self.onnx_session = onnxruntime.InferenceSession(
                onnx_model_file,
                providers=providers
            )
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

    def _get_spec(self, audio: any):
        spec = spectrogram_torch(audio, self.hps.data.filter_length,
                                 self.hps.data.sampling_rate, self.hps.data.hop_length, self.hps.data.win_length,
                                 center=False)
        spec = torch.squeeze(spec, 0)
        return spec

    def generate_input(self, newData: any, inputSize: int, crossfadeSize: int):
        newData = newData.astype(np.float32) / self.hps.data.max_wav_value

        if hasattr(self, "audio_buffer"):
            self.audio_buffer = np.concatenate([self.audio_buffer, newData], 0)  # 過去のデータに連結
        else:
            self.audio_buffer = newData

        convertSize = inputSize + crossfadeSize
        if convertSize < 8192:
            convertSize = 8192
        if convertSize % self.hps.data.hop_length != 0:  # モデルの出力のホップサイズで切り捨てが発生するので補う。
            convertSize = convertSize + (self.hps.data.hop_length - (convertSize % self.hps.data.hop_length))

        self.audio_buffer = self.audio_buffer[-1 * convertSize:]  # 変換対象の部分だけ抽出

        audio = torch.FloatTensor(self.audio_buffer)
        audio_norm = audio.unsqueeze(0)  # unsqueeze
        spec = self._get_spec(audio_norm)
        sid = torch.LongTensor([int(self.settings.srcId)])

        data = (self.text_norm, spec, audio_norm, sid)
        data = TextAudioSpeakerCollate()([data])

        return data

    def _onnx_inference(self, data):
        if hasattr(self, "onnx_session") == False or self.onnx_session == None:
            print("[Voice Changer] No ONNX session.")
            return np.zeros(1).astype(np.int16)

        x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [x for x in data]
        sid_tgt1 = torch.LongTensor([self.settings.dstId])
        # if spec.size()[2] >= 8:
        audio1 = self.onnx_session.run(
            ["audio"],
            {
                "specs": spec.numpy(),
                "lengths": spec_lengths.numpy(),
                "sid_src": sid_src.numpy(),
                "sid_tgt": sid_tgt1.numpy()
            })[0][0, 0] * self.hps.data.max_wav_value
        return audio1

    def _pyTorch_inference(self, data):
        if hasattr(self, "net_g") == False or self.net_g == None:
            print("[Voice Changer] No pyTorch session.")
            return np.zeros(1).astype(np.int16)

        if self.settings.gpu < 0 or self.gpu_num == 0:
            dev = torch.device("cpu")
        else:
            dev = torch.device("cuda", index=self.settings.gpu)

        with torch.no_grad():
            x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [x.to(dev) for x in data]
            sid_target = torch.LongTensor([self.settings.dstId]).to(dev)

            audio1 = (self.net_g.to(dev).voice_conversion(spec, spec_lengths, sid_src=sid_src,
                      sid_tgt=sid_target)[0, 0].data * self.hps.data.max_wav_value)
            result = audio1.float().cpu().numpy()

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
        remove_path = os.path.join("MMVC_Client_v13", "python")
        sys.path = [x for x in sys.path if x.endswith(remove_path) == False]

        for key in list(sys.modules):
            val = sys.modules.get(key)
            try:
                file_path = val.__file__
                if file_path.find("MMVC_Client_v13/python") >= 0:
                    print("remove", key, file_path)
                    sys.modules.pop(key)
            except Exception as e:
                pass
