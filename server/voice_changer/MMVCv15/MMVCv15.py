import sys
sys.path.append("MMVC_Client_v15/python")
from dataclasses import dataclass, asdict
import os
import numpy as np
import torch
import onnxruntime
import pyworld as pw

from models import SynthesizerTrn
from voice_changer.MMVCv15.client_modules import convert_continuos_f0, spectrogram_torch, TextAudioSpeakerCollate, get_hparams_from_file, load_checkpoint

providers = ['OpenVINOExecutionProvider', "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]


@dataclass
class MMVCv15Settings():
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
    def __init__(self):
        self.settings = MMVCv15Settings()
        self.net_g = None
        self.onnx_session = None

        self.gpu_num = torch.cuda.device_count()

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
                requires_grad_dec=self.hps.requires_grad.dec
            )
            self.net_g.eval()
            load_checkpoint(pyTorch_model_file, self.net_g, None)
            # utils.load_checkpoint(pyTorch_model_file, self.net_g, None)

        # ONNXモデル生成
        if onnx_model_file != None:
            ort_options = onnxruntime.SessionOptions()
            ort_options.intra_op_num_threads = 8
            self.onnx_session = onnxruntime.InferenceSession(
                onnx_model_file,
                providers=providers
            )
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

    def _get_f0(self, detector: str, newData: any):

        audio_norm_np = newData.astype(np.float64)
        if detector == "dio":
            _f0, _time = pw.dio(audio_norm_np, self.hps.data.sampling_rate, frame_period=5.5)
            f0 = pw.stonemask(audio_norm_np, _f0, _time, self.hps.data.sampling_rate)
        else:
            f0, t = pw.harvest(audio_norm_np, self.hps.data.sampling_rate, frame_period=5.5, f0_floor=71.0, f0_ceil=1000.0)
        f0 = convert_continuos_f0(f0, int(audio_norm_np.shape[0] / self.hps.data.hop_length))
        f0 = torch.from_numpy(f0.astype(np.float32))
        return f0

    def _get_spec(self, newData: any):
        audio = torch.FloatTensor(newData)
        audio_norm = audio / self.hps.data.max_wav_value  # normalize
        audio_norm = audio_norm.unsqueeze(0)  # unsqueeze
        spec = spectrogram_torch(audio_norm, self.hps.data.filter_length,
                                 self.hps.data.sampling_rate, self.hps.data.hop_length, self.hps.data.win_length,
                                 center=False)
        spec = torch.squeeze(spec, 0)
        return spec

    def generate_input(self, newData: any, convertSize: int):
        newData = newData.astype(np.float32)

        if hasattr(self, "audio_buffer"):
            self.audio_buffer = np.concatenate([self.audio_buffer, newData], 0)  # 過去のデータに連結
        else:
            self.audio_buffer = newData

        self.audio_buffer = self.audio_buffer[-(convertSize):]  # 変換対象の部分だけ抽出

        f0 = self._get_f0(self.settings.f0Detector, self.audio_buffer)  # f0 生成
        spec = self._get_spec(self.audio_buffer)
        sid = torch.LongTensor([int(self.settings.srcId)])

        data = TextAudioSpeakerCollate(
            sample_rate=self.hps.data.sampling_rate,
            hop_size=self.hps.data.hop_length,
            f0_factor=self.settings.f0Factor
        )([(spec, sid, f0)])

        return data

    def _onnx_inference(self, data):
        if hasattr(self, "onnx_session") == False or self.onnx_session == None:
            print("[Voice Changer] No ONNX session.")
            return np.zeros(1).astype(np.int16)

        spec, spec_lengths, sid_src, sin, d = data
        sid_tgt1 = torch.LongTensor([self.settings.dstId])
        audio1 = self.onnx_session.run(
            ["audio"],
            {
                "specs": spec.numpy(),
                "lengths": spec_lengths.numpy(),
                "sin": sin.numpy(),
                "d0": d[0][:1].numpy(),
                "d1": d[1][:1].numpy(),
                "d2": d[2][:1].numpy(),
                "d3": d[3][:1].numpy(),
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
            spec, spec_lengths, sid_src, sin, d = data
            spec = spec.to(dev)
            spec_lengths = spec_lengths.to(dev)
            sid_src = sid_src.to(dev)
            sin = sin.to(dev)
            d = tuple([d[:1].to(dev) for d in d])
            sid_target = torch.LongTensor([self.settings.dstId]).to(dev)

            audio1 = self.net_g.to(dev).voice_conversion(spec, spec_lengths, sin, d, sid_src, sid_target)[0, 0].data * self.hps.data.max_wav_value
            result = audio1.float().cpu().numpy()
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
