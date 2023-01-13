from const import ERROR_NO_ONNX_SESSION
import torch
import math, os, traceback
from scipy.io.wavfile import write, read
import numpy as np
from dataclasses import dataclass, asdict

import onnxruntime


# import utils
# import commons
# from models import SynthesizerTrn

#from text.symbols import symbols
# from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate

# from mel_processing import spectrogram_torch

#from text import text_to_sequence, cleaned_text_to_sequence


################
from symbols import symbols
# from mmvc_client import get_hparams_from_file, load_checkpoint
from models import SynthesizerTrn
################

# from voice_changer.utils import get_hparams_from_file, load_checkpoint
# from voice_changer.models import SynthesizerTrn
# from voice_changer.symbols import symbols

from voice_changer.TrainerFunctions import TextAudioSpeakerCollate, spectrogram_torch, load_checkpoint, get_hparams_from_file

providers = ['OpenVINOExecutionProvider',"CUDAExecutionProvider","DmlExecutionProvider","CPUExecutionProvider"]

@dataclass
class VocieChangerSettings():
    gpu:int = 0
    srcId:int = 107
    dstId:int = 100
    crossFadeOffsetRate:float = 0.1
    crossFadeEndRate:float = 0.9
    crossFadeOverlapRate:float = 0.9
    convertChunkNum:int = 32
    minConvertSize:int = 0
    framework:str = "PyTorch" # PyTorch or ONNX
    pyTorchModelFile:str = ""
    onnxModelFile:str = ""
    configFile:str = ""
    # ↓mutableな物だけ列挙
    intData = ["gpu","srcId", "dstId", "convertChunkNum", "minConvertSize"]
    floatData = [ "crossFadeOffsetRate", "crossFadeEndRate", "crossFadeOverlapRate"]
    strData = ["framework"]

class VoiceChanger():

    def __init__(self, config:str):
        # 初期化
        self.settings = VocieChangerSettings(configFile=config)
        self.unpackedData_length=0
        self.net_g = None
        self.onnx_session = None
        self.currentCrossFadeOffsetRate=0
        self.currentCrossFadeEndRate=0
        self.currentCrossFadeOverlapRate=0
        
        # 共通で使用する情報を収集
        # self.hps = utils.get_hparams_from_file(config)
        self.hps = get_hparams_from_file(config)
        self.gpu_num = torch.cuda.device_count()

        # text_norm = text_to_sequence("a", self.hps.data.text_cleaners)
        # print("text_norm1: ",text_norm)
        # text_norm = commons.intersperse(text_norm, 0)
        # print("text_norm2: ",text_norm)
        # self.text_norm = torch.LongTensor(text_norm)

        self.text_norm = torch.LongTensor([0, 6, 0])        
        self.audio_buffer = torch.zeros(1, 0)
        self.prev_audio = np.zeros(1)
        self.mps_enabled = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

        print(f"VoiceChanger Initialized (GPU_NUM:{self.gpu_num}, mps_enabled:{self.mps_enabled})")

    def loadModel(self, config:str, pyTorch_model_file:str=None, onnx_model_file:str=None):
        self.settings.configFile = config
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
            if data[f]!=None and os.path.exists(data[f]):
                data[f] = os.path.basename(data[f])
            else:
                data[f] = ""

        return data

    def update_setteings(self, key:str, val:any):
        if key == "onnxExecutionProvider" and self.onnx_session != None:
            if val == "CUDAExecutionProvider":
                if self.settings.gpu < 0 or self.settings.gpu >= self.gpu_num:
                    self.settings.gpu = 0
                provider_options=[{'device_id': self.settings.gpu}]
                self.onnx_session.set_providers(providers=[val], provider_options=provider_options)
            else:
                self.onnx_session.set_providers(providers=[val])
        elif key in self.settings.intData:
            setattr(self.settings, key, int(val))
            if key == "gpu" and val >= 0 and val < self.gpu_num and self.onnx_session != None:
                providers = self.onnx_session.get_providers()
                print("Providers:", providers)
                if "CUDAExecutionProvider" in providers:
                    provider_options=[{'device_id': self.settings.gpu}]
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
            cf_end   = int(overlapSize * self.settings.crossFadeEndRate)
            cf_range = cf_end - cf_offset
            percent = np.arange(cf_range) / cf_range

            np_prev_strength = np.cos(percent  * 0.5 * np.pi) ** 2
            np_cur_strength = np.cos((1-percent) * 0.5 * np.pi) ** 2

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
                delattr(self,"prev_audio1")

    def _generate_input(self, unpackedData:any, convertSize:int):
        # 今回変換するデータをテンソルとして整形する
        audio = torch.FloatTensor(unpackedData.astype(np.float32)) # float32でtensorfを作成
        audio_norm = audio / self.hps.data.max_wav_value # normalize
        audio_norm = audio_norm.unsqueeze(0) # unsqueeze
        self.audio_buffer = torch.cat([self.audio_buffer, audio_norm], axis=1) # 過去のデータに連結
        audio_norm = self.audio_buffer[:, -convertSize:] # 変換対象の部分だけ抽出
        self.audio_buffer = audio_norm

        spec = spectrogram_torch(audio_norm, self.hps.data.filter_length,
                                    self.hps.data.sampling_rate, self.hps.data.hop_length, self.hps.data.win_length,
                                    center=False)
        spec = torch.squeeze(spec, 0)
        sid = torch.LongTensor([int(self.settings.srcId)])

        data = (self.text_norm, spec, audio_norm, sid)
        data = TextAudioSpeakerCollate()([data])
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
            })[0][0,0] * self.hps.data.max_wav_value
        if hasattr(self, 'np_prev_audio1') == True:
            overlapSize = int(inputSize * self.settings.crossFadeOverlapRate)
            prev_overlap = self.np_prev_audio1[-1*overlapSize:]
            cur_overlap = audio1[-1*(inputSize + overlapSize) :-1*inputSize]
            # print(prev_overlap.shape, self.np_prev_strength.shape, cur_overlap.shape, self.np_cur_strength.shape)
            # print(">>>>>>>>>>>", -1*(inputSize + overlapSize) , -1*inputSize)
            powered_prev = prev_overlap * self.np_prev_strength
            powered_cur = cur_overlap * self.np_cur_strength
            powered_result = powered_prev + powered_cur

            cur = audio1[-1*inputSize:-1*overlapSize]
            result = np.concatenate([powered_result, cur],axis=0)
        else:
            result = np.zeros(1).astype(np.int16)
        self.np_prev_audio1 = audio1
        return result

    def _pyTorch_inference(self, data, inputSize):
        if hasattr(self, "net_g") == False or self.net_g ==None:
            print("[Voice Changer] No pyTorch session.")
            return np.zeros(1).astype(np.int16)

        if self.settings.gpu < 0 or self.gpu_num == 0:
            with torch.no_grad():
                x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [x.cpu() for x in data]
                sid_tgt1 = torch.LongTensor([self.settings.dstId]).cpu()
                audio1 = (self.net_g.cpu().voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt1)[0, 0].data * self.hps.data.max_wav_value)

                if self.prev_strength.device != torch.device('cpu'):
                    print(f"prev_strength move from {self.prev_strength.device} to cpu")
                    self.prev_strength = self.prev_strength.cpu()
                if self.cur_strength.device != torch.device('cpu'):
                    print(f"cur_strength move from {self.cur_strength.device} to cpu")
                    self.cur_strength = self.cur_strength.cpu()

                if hasattr(self, 'prev_audio1') == True and self.prev_audio1.device == torch.device('cpu'): # prev_audio1が所望のデバイスに無い場合は一回休み。
                    overlapSize = int(inputSize * self.settings.crossFadeOverlapRate)
                    prev_overlap = self.prev_audio1[-1*overlapSize:]
                    cur_overlap = audio1[-1*(inputSize + overlapSize) :-1*inputSize]
                    powered_prev = prev_overlap * self.prev_strength
                    powered_cur = cur_overlap * self.cur_strength
                    powered_result = powered_prev + powered_cur

                    cur = audio1[-1*inputSize:-1*overlapSize] # 今回のインプットの生部分。(インプット - 次回のCrossfade部分)。
                    result = torch.cat([powered_result, cur],axis=0) # Crossfadeと今回のインプットの生部分を結合

                else:
                    cur = audio1[-2*inputSize:-1*inputSize]
                    result = cur

                self.prev_audio1 = audio1
                result = result.cpu().float().numpy()

        else:
            with torch.no_grad():
                x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [x.cuda(self.settings.gpu) for x in data]
                sid_tgt1 = torch.LongTensor([self.settings.dstId]).cuda(self.settings.gpu)
                audio1 = self.net_g.cuda(self.settings.gpu).voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt1)[0, 0].data * self.hps.data.max_wav_value

                if self.prev_strength.device != torch.device('cuda', self.settings.gpu):
                    print(f"prev_strength move from {self.prev_strength.device} to gpu{self.settings.gpu}")
                    self.prev_strength = self.prev_strength.cuda(self.settings.gpu)
                if self.cur_strength.device != torch.device('cuda', self.settings.gpu):
                    print(f"cur_strength move from {self.cur_strength.device} to gpu{self.settings.gpu}")
                    self.cur_strength = self.cur_strength.cuda(self.settings.gpu)



                if hasattr(self, 'prev_audio1') == True and self.prev_audio1.device == torch.device('cuda', self.settings.gpu):
                    overlapSize = int(inputSize * self.settings.crossFadeOverlapRate)
                    prev_overlap = self.prev_audio1[-1*overlapSize:]
                    cur_overlap = audio1[-1*(inputSize + overlapSize) :-1*inputSize]
                    powered_prev = prev_overlap * self.prev_strength
                    powered_cur = cur_overlap * self.cur_strength
                    powered_result = powered_prev + powered_cur

                    cur = audio1[-1*inputSize:-1*overlapSize] # 今回のインプットの生部分。(インプット - 次回のCrossfade部分)。
                    result = torch.cat([powered_result, cur],axis=0) # Crossfadeと今回のインプットの生部分を結合

                else:
                    cur = audio1[-2*inputSize:-1*inputSize]
                    result = cur
                self.prev_audio1 = audio1

                result = result.cpu().float().numpy()
        return result
            

    def on_request(self,  unpackedData:any):
        convertSize = self.settings.convertChunkNum * 128 # 128sample/1chunk

        if unpackedData.shape[0]*(1 + self.settings.crossFadeOverlapRate) + 1024 > convertSize:
            convertSize = int(unpackedData.shape[0]*(1 + self.settings.crossFadeOverlapRate)) + 1024
        if convertSize < self.settings.minConvertSize:
            convertSize = self.settings.minConvertSize
        # print("convert Size", unpackedData.shape[0], unpackedData.shape[0]*(1 + self.settings.crossFadeOverlapRate), convertSize, self.settings.minConvertSize)

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

