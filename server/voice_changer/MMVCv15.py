import sys
sys.path.append("MMVC_Client/python")
import os
from dataclasses import dataclass, asdict

import numpy as np
import torch
import onnxruntime
import pyworld as pw

from voice_changer.client_modules import convert_continuos_f0, spectrogram_torch, TextAudioSpeakerCollate, get_hparams_from_file, load_checkpoint
from models import SynthesizerTrn

from const import ERROR_NO_ONNX_SESSION, TMP_DIR


providers = ['OpenVINOExecutionProvider', "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]


@dataclass
class MMVCv15Settings():
    gpu: int = 0
    srcId: int = 0
    dstId: int = 101

    inputSampleRate: int = 24000  # 48000 or 24000

    crossFadeOffsetRate: float = 0.1
    crossFadeEndRate: float = 0.9
    crossFadeOverlapSize: int = 4096

    f0Factor: float = 1.0
    f0Detector: str = "dio"  # dio or harvest
    recordIO: int = 0  # 0:off, 1:on

    framework: str = "PyTorch"  # PyTorch or ONNX
    pyTorchModelFile: str = ""
    onnxModelFile: str = ""
    configFile: str = ""

    # ↓mutableな物だけ列挙
    intData = ["gpu", "srcId", "dstId", "inputSampleRate", "crossFadeOverlapSize", "recordIO"]
    floatData = ["crossFadeOffsetRate", "crossFadeEndRate", "f0Factor"]
    strData = ["framework", "f0Detector"]


class MMVCv15:
    def __init__(self):
        # 初期化
        self.settings = MMVCv15Settings()
        self.net_g = None
        self.onnx_session = None

        self.gpu_num = torch.cuda.device_count()
        self.text_norm = torch.LongTensor([0, 6, 0])
        self.audio_buffer = torch.zeros(1, 0)
        self.mps_enabled = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

        print(f"VoiceChanger Initialized (GPU_NUM:{self.gpu_num}, mps_enabled:{self.mps_enabled})")

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

        # ONNXモデル生成
        if onnx_model_file != None:
            ort_options = onnxruntime.SessionOptions()
            ort_options.intra_op_num_threads = 8
            self.onnx_session = onnxruntime.InferenceSession(
                onnx_model_file,
                providers=providers
            )
        return self.get_info()

    def destroy(self):
        del self.net_g
        del self.onnx_session

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
            if key == "crossFadeOffsetRate" or key == "crossFadeEndRate":
                self.unpackedData_length = 0

        elif key in self.settings.floatData:
            setattr(self.settings, key, float(val))
        elif key in self.settings.strData:
            setattr(self.settings, key, str(val))
        else:
            print(f"{key} is not mutalbe variable!")

        return self.get_info()

    def _generate_input(self, unpackedData: any, convertSize: int):
        # 今回変換するデータをテンソルとして整形する
        audio = torch.FloatTensor(unpackedData.astype(np.float32))  # float32でtensorfを作成
        audio_norm = audio / self.hps.data.max_wav_value  # normalize
        audio_norm = audio_norm.unsqueeze(0)  # unsqueeze
        self.audio_buffer = torch.cat([self.audio_buffer, audio_norm], axis=1)  # 過去のデータに連結
        # audio_norm = self.audio_buffer[:, -(convertSize + 1280 * 2):]  # 変換対象の部分だけ抽出
        audio_norm = self.audio_buffer[:, -(convertSize):]  # 変換対象の部分だけ抽出
        self.audio_buffer = audio_norm

        # TBD: numpy <--> pytorch変換が行ったり来たりしているが、まずは動かすことを最優先。
        audio_norm_np = audio_norm.squeeze().numpy().astype(np.float64)
        if self.settings.f0Detector == "dio":
            _f0, _time = pw.dio(audio_norm_np, self.hps.data.sampling_rate, frame_period=5.5)
            f0 = pw.stonemask(audio_norm_np, _f0, _time, self.hps.data.sampling_rate)
        else:
            f0, t = pw.harvest(audio_norm_np, self.hps.data.sampling_rate, frame_period=5.5, f0_floor=71.0, f0_ceil=1000.0)
        f0 = convert_continuos_f0(f0, int(audio_norm_np.shape[0] / self.hps.data.hop_length))
        f0 = torch.from_numpy(f0.astype(np.float32))

        spec = spectrogram_torch(audio_norm, self.hps.data.filter_length,
                                 self.hps.data.sampling_rate, self.hps.data.hop_length, self.hps.data.win_length,
                                 center=False)
        # dispose_stft_specs = 2
        # spec = spec[:, dispose_stft_specs:-dispose_stft_specs]
        # f0 = f0[dispose_stft_specs:-dispose_stft_specs]
        spec = torch.squeeze(spec, 0)
        sid = torch.LongTensor([int(self.settings.srcId)])

        # data = (self.text_norm, spec, audio_norm, sid)
        # data = TextAudioSpeakerCollate()([data])
        data = TextAudioSpeakerCollate(
            sample_rate=self.hps.data.sampling_rate,
            hop_size=self.hps.data.hop_length,
            f0_factor=self.settings.f0Factor
        )([(spec, sid, f0)])

        return data

    def _onnx_inference(self, data, inputSize):
        if hasattr(self, "onnx_session") == False or self.onnx_session == None:
            print("[Voice Changer] No ONNX session.")
            return np.zeros(1).astype(np.int16)

        # x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [x for x in data]
        # sid_tgt1 = torch.LongTensor([self.settings.dstId])

        spec, spec_lengths, sid_src, sin, d = data
        sid_tgt1 = torch.LongTensor([self.settings.dstId])
        # if spec.size()[2] >= 8:
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

        if hasattr(self, 'np_prev_audio1') == True:
            overlapSize = min(self.settings.crossFadeOverlapSize, inputSize)
            prev_overlap = self.np_prev_audio1[-1 * overlapSize:]
            cur_overlap = audio1[-1 * (inputSize + overlapSize):-1 * inputSize]
            # print(prev_overlap.shape, self.np_prev_strength.shape, cur_overlap.shape, self.np_cur_strength.shape)
            # print(">>>>>>>>>>>", -1*(inputSize + overlapSize) , -1*inputSize)
            powered_prev = prev_overlap * self.np_prev_strength
            powered_cur = cur_overlap * self.np_cur_strength
            powered_result = powered_prev + powered_cur

            cur = audio1[-1 * inputSize:-1 * overlapSize]
            result = np.concatenate([powered_result, cur], axis=0)
        else:
            result = np.zeros(1).astype(np.int16)
        self.np_prev_audio1 = audio1
        return result

    def _pyTorch_inference(self, data, inputSize):
        if hasattr(self, "net_g") == False or self.net_g == None:
            print("[Voice Changer] No pyTorch session.")
            return np.zeros(1).astype(np.int16)

        if self.settings.gpu < 0 or self.gpu_num == 0:
            with torch.no_grad():
                spec, spec_lengths, sid_src, sin, d = data
                spec = spec.cpu()
                spec_lengths = spec_lengths.cpu()
                sid_src = sid_src.cpu()
                sin = sin.cpu()
                d = tuple([d[:1].cpu() for d in d])
                sid_target = torch.LongTensor([self.settings.dstId]).cpu()

                audio1 = self.net_g.cpu().voice_conversion(spec, spec_lengths, sin, d, sid_src, sid_target)[0, 0].data * self.hps.data.max_wav_value

                if self.prev_strength.device != torch.device('cpu'):
                    print(f"prev_strength move from {self.prev_strength.device} to cpu")
                    self.prev_strength = self.prev_strength.cpu()
                if self.cur_strength.device != torch.device('cpu'):
                    print(f"cur_strength move from {self.cur_strength.device} to cpu")
                    self.cur_strength = self.cur_strength.cpu()

                if hasattr(self, 'prev_audio1') == True and self.prev_audio1.device == torch.device('cpu'):  # prev_audio1が所望のデバイスに無い場合は一回休み。
                    overlapSize = min(self.settings.crossFadeOverlapSize, inputSize)
                    prev_overlap = self.prev_audio1[-1 * overlapSize:]
                    cur_overlap = audio1[-1 * (inputSize + overlapSize):-1 * inputSize]
                    powered_prev = prev_overlap * self.prev_strength
                    powered_cur = cur_overlap * self.cur_strength
                    powered_result = powered_prev + powered_cur

                    cur = audio1[-1 * inputSize:-1 * overlapSize]  # 今回のインプットの生部分。(インプット - 次回のCrossfade部分)。
                    result = torch.cat([powered_result, cur], axis=0)  # Crossfadeと今回のインプットの生部分を結合

                else:
                    cur = audio1[-2 * inputSize:-1 * inputSize]
                    result = cur

                self.prev_audio1 = audio1
                result = result.cpu().float().numpy()

        else:
            with torch.no_grad():
                spec, spec_lengths, sid_src, sin, d = data
                spec = spec.cuda(self.settings.gpu)
                spec_lengths = spec_lengths.cuda(self.settings.gpu)
                sid_src = sid_src.cuda(self.settings.gpu)
                sin = sin.cuda(self.settings.gpu)
                d = tuple([d[:1].cuda(self.settings.gpu) for d in d])
                sid_target = torch.LongTensor([self.settings.dstId]).cuda(self.settings.gpu)

                # audio1 = self.net_g.cuda(self.settings.gpu).voice_conversion(spec, spec_lengths, sid_src=sid_src,
                #  sid_tgt=sid_tgt1)[0, 0].data * self.hps.data.max_wav_value

                audio1 = self.net_g.cuda(self.settings.gpu).voice_conversion(spec, spec_lengths, sin, d,
                                                                             sid_src, sid_target)[0, 0].data * self.hps.data.max_wav_value

                if self.prev_strength.device != torch.device('cuda', self.settings.gpu):
                    print(f"prev_strength move from {self.prev_strength.device} to gpu{self.settings.gpu}")
                    self.prev_strength = self.prev_strength.cuda(self.settings.gpu)
                if self.cur_strength.device != torch.device('cuda', self.settings.gpu):
                    print(f"cur_strength move from {self.cur_strength.device} to gpu{self.settings.gpu}")
                    self.cur_strength = self.cur_strength.cuda(self.settings.gpu)

                if hasattr(self, 'prev_audio1') == True and self.prev_audio1.device == torch.device('cuda', self.settings.gpu):
                    overlapSize = min(self.settings.crossFadeOverlapSize, inputSize)
                    prev_overlap = self.prev_audio1[-1 * overlapSize:]
                    cur_overlap = audio1[-1 * (inputSize + overlapSize):-1 * inputSize]
                    powered_prev = prev_overlap * self.prev_strength
                    powered_cur = cur_overlap * self.cur_strength
                    powered_result = powered_prev + powered_cur

                    # print(overlapSize, prev_overlap.shape, cur_overlap.shape, self.prev_strength.shape, self.cur_strength.shape)
                    # print(self.prev_audio1.shape, audio1.shape, inputSize, overlapSize)

                    cur = audio1[-1 * inputSize:-1 * overlapSize]  # 今回のインプットの生部分。(インプット - 次回のCrossfade部分)。
                    result = torch.cat([powered_result, cur], axis=0)  # Crossfadeと今回のインプットの生部分を結合

                else:
                    cur = audio1[-2 * inputSize:-1 * inputSize]
                    result = cur
                self.prev_audio1 = audio1

                result = result.cpu().float().numpy()
        return result
