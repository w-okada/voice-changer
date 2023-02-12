from const import ERROR_NO_ONNX_SESSION
import torch
import os
import traceback
import numpy as np
from dataclasses import dataclass, asdict

import onnxruntime

from symbols import symbols
from models import SynthesizerTrn

import pyworld as pw

# from voice_changer.TrainerFunctions import TextAudioSpeakerCollate, spectrogram_torch, load_checkpoint, get_hparams_from_file

from voice_changer.client_modules import convert_continuos_f0, spectrogram_torch, TextAudioSpeakerCollate, get_hparams_from_file, load_checkpoint


providers = ['OpenVINOExecutionProvider', "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]


import wave


class MockStream:
    """
    オーディオストリーミング入出力をファイル入出力にそのまま置き換えるためのモック
    """

    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate
        self.start_count = 2
        self.end_count = 2
        self.fr = None
        self.fw = None

    def open_inputfile(self, input_filename):
        self.fr = wave.open(input_filename, 'rb')

    def open_outputfile(self, output_filename):
        self.fw = wave.open(output_filename, 'wb')
        self.fw.setnchannels(1)
        self.fw.setsampwidth(2)
        self.fw.setframerate(self.sampling_rate)

    def read(self, length, exception_on_overflow=False):
        if self.start_count > 0:
            wav = bytes(length * 2)
            self.start_count -= 1  # 最初の2回はダミーの空データ送る
        else:
            wav = self.fr.readframes(length)
        if len(wav) <= 0:  # データなくなってから最後の2回はダミーの空データを送る
            wav = bytes(length * 2)
            self.end_count -= 1
            if self.end_count < 0:
                Hyperparameters.VC_END_FLAG = True
        return wav

    def write(self, wav):
        self.fw.writeframes(wav)

    def stop_stream(self):
        pass

    def close(self):
        if self.fr != None:
            self.fr.close()
            self.fr = None
        if self.fw != None:
            self.fw.close()
            self.fw = None


@dataclass
class VocieChangerSettings():
    gpu: int = 0
    srcId: int = 107
    dstId: int = 100
    crossFadeOffsetRate: float = 0.1
    crossFadeEndRate: float = 0.9
    crossFadeOverlapRate: float = 0.9
    convertChunkNum: int = 32
    minConvertSize: int = 0
    framework: str = "ONNX"  # PyTorch or ONNX
    f0Factor: float = 1.0

    pyTorchModelFile: str = ""
    onnxModelFile: str = ""
    configFile: str = ""

    # ↓mutableな物だけ列挙
    intData = ["gpu", "srcId", "dstId", "convertChunkNum", "minConvertSize"]
    floatData = ["crossFadeOffsetRate", "crossFadeEndRate", "crossFadeOverlapRate", "f0Factor"]
    strData = ["framework"]


class VoiceChanger():

    def __init__(self):
        # 初期化
        self.settings = VocieChangerSettings()
        self.unpackedData_length = 0
        self.net_g = None
        self.onnx_session = None
        self.currentCrossFadeOffsetRate = 0
        self.currentCrossFadeEndRate = 0
        self.currentCrossFadeOverlapRate = 0

        self.gpu_num = torch.cuda.device_count()
        self.text_norm = torch.LongTensor([0, 6, 0])
        self.audio_buffer = torch.zeros(1, 0)
        self.prev_audio = np.zeros(1)
        self.mps_enabled = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

        mock_stream = MockStream(24000)
        mock_stream.open_outputfile("out.wav")
        self.out = mock_stream
        mock_stream_in = MockStream(24000)
        mock_stream_in.open_outputfile("in.wav")
        self.stream_in = mock_stream_in

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

    def destroy(self):
        del self.net_g
        del self.onnx_session

    def get_info(self):
        data = asdict(self.settings)

        data["onnxExecutionProvider"] = self.onnx_session.get_providers() if self.onnx_session != None else []
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

    def _generate_strength(self, unpackedData):

        if self.unpackedData_length != unpackedData.shape[0] or self.currentCrossFadeOffsetRate != self.settings.crossFadeOffsetRate or self.currentCrossFadeEndRate != self.settings.crossFadeEndRate or self.currentCrossFadeOverlapRate != self.settings.crossFadeOverlapRate:
            self.unpackedData_length = unpackedData.shape[0]
            self.currentCrossFadeOffsetRate = self.settings.crossFadeOffsetRate
            self.currentCrossFadeEndRate = self.settings.crossFadeEndRate
            self.currentCrossFadeOverlapRate = self.settings.crossFadeOverlapRate

            overlapSize = int(unpackedData.shape[0] * self.settings.crossFadeOverlapRate)

            cf_offset = int(overlapSize * self.settings.crossFadeOffsetRate)
            cf_end = int(overlapSize * self.settings.crossFadeEndRate)
            cf_range = cf_end - cf_offset
            percent = np.arange(cf_range) / cf_range

            np_prev_strength = np.cos(percent * 0.5 * np.pi) ** 2
            np_cur_strength = np.cos((1 - percent) * 0.5 * np.pi) ** 2

            self.np_prev_strength = np.concatenate([np.ones(cf_offset), np_prev_strength, np.zeros(overlapSize - cf_offset - len(np_prev_strength))])
            self.np_cur_strength = np.concatenate([np.zeros(cf_offset), np_cur_strength, np.ones(overlapSize - cf_offset - len(np_cur_strength))])

            self.prev_strength = torch.FloatTensor(self.np_prev_strength)
            self.cur_strength = torch.FloatTensor(self.np_cur_strength)

            # torch.set_printoptions(edgeitems=2100)
            print("Generated Strengths")
            # print(f"cross fade: start:{cf_offset} end:{cf_end} range:{cf_range}")
            # print(f"target_len:{unpackedData.shape[0]}, prev_len:{len(self.prev_strength)} cur_len:{len(self.cur_strength)}")
            # print("Prev", self.prev_strength)
            # print("Cur", self.cur_strength)

            # ひとつ前の結果とサイズが変わるため、記録は消去する。
            if hasattr(self, 'prev_audio1') == True:
                delattr(self, "prev_audio1")

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
        _f0, _time = pw.dio(audio_norm_np, self.hps.data.sampling_rate, frame_period=5.5)
        f0 = pw.stonemask(audio_norm_np, _f0, _time, self.hps.data.sampling_rate)
        # print("type:", audio_norm_np.dtype)
        # f0, t = pw.harvest(audio_norm_np, self.hps.data.sampling_rate, frame_period=5.5, f0_floor=71.0, f0_ceil=1000.0)
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
        if hasattr(self, 'np_prev_audio1') == True:
            overlapSize = int(inputSize * self.settings.crossFadeOverlapRate)
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
                    overlapSize = int(inputSize * self.settings.crossFadeOverlapRate)
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
                    overlapSize = int(inputSize * self.settings.crossFadeOverlapRate)
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
        return result

    def on_request_(self, unpackedData: any):
        convertSize = self.settings.convertChunkNum * 128  # 128sample/1chunk
        self.stream_in.write(unpackedData.astype(np.int16).tobytes())
        # print("convsize:", unpackedData.shape[0] * (1 + self.settings.crossFadeOverlapRate))
        if unpackedData.shape[0] * (1 + self.settings.crossFadeOverlapRate) + 1024 > convertSize:
            convertSize = int(unpackedData.shape[0] * (1 + self.settings.crossFadeOverlapRate)) + 1024
        if convertSize < self.settings.minConvertSize:
            convertSize = self.settings.minConvertSize
        # print("convert Size", unpackedData.shape[0], unpackedData.shape[0]*(1 + self.settings.crossFadeOverlapRate), convertSize, self.settings.minConvertSize)

        # convertSize = 8192

        self._generate_strength(unpackedData)
        data = self._generate_input(unpackedData, convertSize)

        try:
            if self.settings.framework == "ONNX":
                result = self._onnx_inference(data, unpackedData.shape[0])
            else:
                result = self._pyTorch_inference(data, unpackedData.shape[0])

        except Exception as e:
            print("VC PROCESSING!!!! EXCEPTION!!!", e)
            print(traceback.format_exc())
            if hasattr(self, "np_prev_audio1"):
                del self.np_prev_audio1
            if hasattr(self, "prev_audio1"):
                del self.prev_audio1
            return np.zeros(1).astype(np.int16)

        result = result.astype(np.int16)
        # print("on_request result size:",result.shape)
        return result


#########################################################################################


    def overlap_merge(self, now_wav, prev_wav, overlap_length):
        """
        生成したwavデータを前回生成したwavデータとoverlap_lengthだけ重ねてグラデーション的にマージします
        終端のoverlap_lengthぶんは次回マージしてから再生するので削除します

        Parameters
        ----------
        now_wav: 今回生成した音声wavデータ
        prev_wav: 前回生成した音声wavデータ
        overlap_length: 重ねる長さ
        """
        if overlap_length == 0:
            return now_wav
        gradation = np.arange(overlap_length) / overlap_length
        now = np.frombuffer(now_wav, dtype='int16')
        prev = np.frombuffer(prev_wav, dtype='int16')
        now_head = now[:overlap_length]
        prev_tail = prev[-overlap_length:]
        print("merge params:", gradation.shape, now.shape, prev.shape, now_head.shape, prev_tail.shape)
        merged = prev_tail * (np.cos(gradation * np.pi * 0.5) ** 2) + now_head * (np.cos((1 - gradation) * np.pi * 0.5) ** 2)
        # merged = prev_tail * (1 - gradation) + now_head * gradation
        overlapped = np.append(merged, now[overlap_length:-overlap_length])
        signal = np.round(overlapped, decimals=0)
        signal = signal.astype(np.int16)
        # signal = signal.astype(np.int16).tobytes()
        return signal

    def on_request(self, unpackedData: any):

        self._generate_strength(unpackedData)

        convertSize = 8192
        unpackedData = unpackedData.astype(np.int16)
        if hasattr(self, 'stored_raw_input') == False:
            self.stored_raw_input = unpackedData
        else:
            self.stored_raw_input = np.concatenate([self.stored_raw_input, unpackedData])

        self.stored_raw_input = self.stored_raw_input[-1 * (convertSize):]
        processing_input = self.stored_raw_input

        print("signal_shape1", unpackedData.shape, processing_input.shape, processing_input.dtype)
        processing_input = processing_input / self.hps.data.max_wav_value
        print("type:", processing_input.dtype)
        _f0, _time = pw.dio(processing_input, self.hps.data.sampling_rate, frame_period=5.5)
        f0 = pw.stonemask(processing_input, _f0, _time, self.hps.data.sampling_rate)
        f0 = convert_continuos_f0(f0, int(processing_input.shape[0] / self.hps.data.hop_length))
        f0 = torch.from_numpy(f0.astype(np.float32))

        print("signal_shape2", f0.shape)

        processing_input = torch.from_numpy(processing_input.astype(np.float32)).clone()
        with torch.no_grad():
            trans_length = processing_input.size()[0]
            # spec, sid = get_audio_text_speaker_pair(signal.view(1, trans_length), Hyperparameters.SOURCE_ID)
            processing_input_v = processing_input.view(1, trans_length)  # unsqueezeと同じ

            print("processing_input_v shape:", processing_input_v.shape)
            spec = spectrogram_torch(processing_input_v, self.hps.data.filter_length,
                                     self.hps.data.sampling_rate, self.hps.data.hop_length, self.hps.data.win_length,
                                     center=False)
            spec = torch.squeeze(spec, 0)
            sid = torch.LongTensor([int(self.settings.srcId)])
            dispose_stft_specs = 2
            spec = spec[:, dispose_stft_specs:-dispose_stft_specs]
            f0 = f0[dispose_stft_specs:-dispose_stft_specs]
            print("spec shape:", spec.shape)
            data = TextAudioSpeakerCollate(
                sample_rate=self.hps.data.sampling_rate,
                hop_size=self.hps.data.hop_length,
                f0_factor=self.settings.f0Factor
            )([(spec, sid, f0)])

            if self.settings.gpu >= 0 or self.gpu_num > 0:
                # spec, spec_lengths, sid_src, sin, d = [x.cuda(Hyperparameters.GPU_ID) for x in data]
                spec, spec_lengths, sid_src, sin, d = data
                spec = spec.cuda(self.settings.gpu)
                spec_lengths = spec_lengths.cuda(self.settings.gpu)
                sid_src = sid_src.cuda(self.settings.gpu)
                sin = sin.cuda(self.settings.gpu)
                d = tuple([d[:1].cuda(self.settings.gpu) for d in d])
                sid_target = torch.LongTensor([self.settings.dstId]).cuda(self.settings.gpu)
                audio = self.net_g.cuda(self.settings.gpu).voice_conversion(spec, spec_lengths,
                                                                            sin, d, sid_src, sid_target)[0, 0].data.cpu().float().numpy()
            else:
                spec, spec_lengths, sid_src, sin, d = data
                sid_target = torch.LongTensor([self.settings.dstId])
                audio = self.net_g.voice_conversion(spec, spec_lengths, sin, d, sid_src, sid_target)[0, 0].data.cpu().float().numpy()

            dispose_conv1d_length = 1280
            audio = audio[dispose_conv1d_length:-dispose_conv1d_length]
            audio = audio * self.hps.data.max_wav_value
            audio = audio.astype(np.int16)
            print("fin audio shape:", audio.shape)
            audio = audio.tobytes()

            if hasattr(self, "prev_audio"):
                try:
                    audio1 = self.overlap_merge(audio, self.prev_audio, 1024)
                except:
                    audio1 = np.zeros(1).astype(np.int16)
                    pass
                    # return np.zeros(1).astype(np.int16)
            else:
                audio1 = np.zeros(1).astype(np.int16)

            self.prev_audio = audio
            self.out.write(audio)
            self.stream_in.write(unpackedData.tobytes())

        return audio1
