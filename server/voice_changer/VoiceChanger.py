from const import ERROR_NO_ONNX_SESSION
import torch
import math, os, traceback
from scipy.io.wavfile import write, read
import numpy as np
from dataclasses import dataclass, asdict
import utils
import commons
from models import SynthesizerTrn

from text.symbols import symbols
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate

from mel_processing import spectrogram_torch
from text import text_to_sequence, cleaned_text_to_sequence
import onnxruntime

providers = ['OpenVINOExecutionProvider',"CUDAExecutionProvider","DmlExecutionProvider","CPUExecutionProvider"]

@dataclass
class VocieChangerSettings():
    gpu:int = 0
    srcId:int = 107
    dstId:int = 100
    crossFadeOffsetRate:float = 0.1
    crossFadeEndRate:float = 0.9
    convertChunkNum:int = 32
    framework:str = "PyTorch" # PyTorch or ONNX
    pyTorch_model_file:str = ""
    onnx_model_file:str = ""
    config_file:str = ""
    # ↓mutableな物だけ列挙
    intData = ["gpu","srcId", "dstId", "convertChunkNum"]
    floatData = [ "crossFadeOffsetRate", "crossFadeEndRate",]
    strData = ["framework"]

class VoiceChanger():

    def __init__(self, config:str):
        # 初期化
        self.settings = VocieChangerSettings(config_file=config)
        self.unpackedData_length=0
        # 共通で使用する情報を収集
        self.hps = utils.get_hparams_from_file(config)
        self.gpu_num = torch.cuda.device_count()

        text_norm = text_to_sequence("a", self.hps.data.text_cleaners)
        text_norm = commons.intersperse(text_norm, 0)
        self.text_norm = torch.LongTensor(text_norm)
        self.audio_buffer = torch.zeros(1, 0)
        self.prev_audio = np.zeros(1)
        self.mps_enabled = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

        print(f"VoiceChanger Initialized (GPU_NUM:{self.gpu_num}, mps_enabled:{self.mps_enabled})")

    def loadModel(self, config:str, pyTorch_model_file:str=None, onnx_model_file:str=None):
        self.settings.config_file = config
        self.settings.pyTorch_model_file = pyTorch_model_file
        self.settings.onnx_model_file = onnx_model_file
        
        # PyTorchモデル生成
        if pyTorch_model_file != None:
            self.net_g = SynthesizerTrn(
                len(symbols),
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers=self.hps.data.n_speakers,
                **self.hps.model)
            self.net_g.eval()
            utils.load_checkpoint(pyTorch_model_file, self.net_g, None)
        else:
            self.net_g = None

        # ONNXモデル生成
        if onnx_model_file != None:
            ort_options = onnxruntime.SessionOptions()
            ort_options.intra_op_num_threads = 8
            self.onnx_session = onnxruntime.InferenceSession(
                onnx_model_file,
                providers=providers
            )
        else:
            self.onnx_session = None


    def destroy(self):
        if hasattr(self, "net_g"):
            del self.net_g
        if hasattr(self, "onnx_session"):
            del self.onnx_session    

    def get_info(self):
        data = asdict(self.settings)
        data["providers"] = self.onnx_session.get_providers() if self.onnx_session != None else ""
        files = ["config_file", "pyTorch_model_file", "onnx_model_file"]
        for f in files:
            if os.path.exists(f):
                data[f] = os.path.basename(data[f])
            else:
                data[f] = ""

        return data

    def update_setteings(self, key:str, val:any):
        if key == "onnxExecutionProvider" and self.onnx_session != None:
            if val == "CUDAExecutionProvider":
                provider_options=[{'device_id': self.settings.gpu}]
                self.onnx_session.set_providers(providers=[val], provider_options=provider_options)
            else:
                self.onnx_session.set_providers(providers=[val])
            return self.get_info()
        elif key in self.settings.intData:
            setattr(self.settings, key, int(val))
            if key == "gpu" and val >= 0 and val < self.gpu_num and self.onnx_session != None:
                providers = self.onnx_session.get_providers()
                print("Providers::::", providers)
                if "CUDAExecutionProvider" in providers:
                    provider_options=[{'device_id': self.settings.gpu}]
                    self.onnx_session.set_providers(providers=["CUDAExecutionProvider"], provider_options=provider_options)
            if key == "crossFadeOffsetRate" or key == "crossFadeEndRate":
                self.unpackedData_length = 0
            return self.get_info()
        elif key in self.settings.floatData:
            setattr(self.settings, key, float(val))
            return self.get_info()
        elif key in self.settings.strData:
            setattr(self.settings, key, str(val))
            return self.get_info()
        else:
            print(f"{key} is not mutalbe variable!")
            return self.get_info()



    def _generate_strength(self, unpackedData):

        if self.unpackedData_length != unpackedData.shape[0]:
            self.unpackedData_length = unpackedData.shape[0]
            cf_offset = int(unpackedData.shape[0] * self.settings.crossFadeOffsetRate)
            cf_end   = int(unpackedData.shape[0] * self.settings.crossFadeEndRate)
            cf_range = cf_end - cf_offset
            percent = np.arange(cf_range) / cf_range

            np_prev_strength = np.cos(percent  * 0.5 * np.pi) ** 2
            np_cur_strength = np.cos((1-percent) * 0.5 * np.pi) ** 2

            self.np_prev_strength = np.concatenate([np.ones(cf_offset), np_prev_strength, np.zeros(unpackedData.shape[0]-cf_offset-len(np_prev_strength))])
            self.np_cur_strength = np.concatenate([np.zeros(cf_offset), np_cur_strength, np.ones(unpackedData.shape[0]-cf_offset-len(np_cur_strength))])

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
            prev = self.np_prev_audio1[-1*inputSize:]
            cur  = audio1[-2*inputSize:-1*inputSize]
            # print(prev.shape, self.np_prev_strength.shape, cur.shape, self.np_cur_strength.shape)
            powered_prev = prev * self.np_prev_strength
            powered_cur = cur * self.np_cur_strength
            result = powered_prev + powered_cur
            #result = prev * self.np_prev_strength + cur * self.np_cur_strength
        else:
            cur = audio1[-2*inputSize:-1*inputSize]
            result = cur
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
                audio1 = (self.net_g.cpu().voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt1)[0][0, 0].data * self.hps.data.max_wav_value)

                if self.prev_strength.device != torch.device('cpu'):
                    print(f"prev_strength move from {self.prev_strength.device} to cpu")
                    self.prev_strength = self.prev_strength.cpu()
                if self.cur_strength.device != torch.device('cpu'):
                    print(f"cur_strength move from {self.cur_strength.device} to cpu")
                    self.cur_strength = self.cur_strength.cpu()

                if hasattr(self, 'prev_audio1') == True and self.prev_audio1.device == torch.device('cpu'):
                    prev = self.prev_audio1[-1*inputSize:]
                    cur  = audio1[-2*inputSize:-1*inputSize]
                    result = prev * self.prev_strength + cur * self.cur_strength
                else:
                    cur = audio1[-2*inputSize:-1*inputSize]
                    result = cur

                self.prev_audio1 = audio1
                result = result.cpu().float().numpy()

        else:
            with torch.no_grad():
                x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [x.cuda(self.settings.gpu) for x in data]
                sid_tgt1 = torch.LongTensor([self.settings.dstId]).cuda(self.settings.gpu)
                audio1 = self.net_g.cuda(self.settings.gpu).voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt1)[0][0, 0].data * self.hps.data.max_wav_value

                if self.prev_strength.device != torch.device('cuda', self.settings.gpu):
                    print(f"prev_strength move from {self.prev_strength.device} to gpu{self.settings.gpu}")
                    self.prev_strength = self.prev_strength.cuda(self.settings.gpu)
                if self.cur_strength.device != torch.device('cuda', self.settings.gpu):
                    print(f"cur_strength move from {self.cur_strength.device} to gpu{self.settings.gpu}")
                    self.cur_strength = self.cur_strength.cuda(self.settings.gpu)



                if hasattr(self, 'prev_audio1') == True and self.prev_audio1.device == torch.device('cuda', self.settings.gpu):
                    prev = self.prev_audio1[-1*inputSize:]
                    cur  = audio1[-2*inputSize:-1*inputSize]
                    result = prev * self.prev_strength + cur * self.cur_strength
                    # print("merging...", prev.shape, cur.shape)
                else:
                    cur = audio1[-2*inputSize:-1*inputSize]
                    result = cur
                    # print("no merging...", cur.shape)
                self.prev_audio1 = audio1

                #print(result)                    
                result = result.cpu().float().numpy()
        return result
            

    def on_request(self,  unpackedData:any):
        convertSize = self.settings.convertChunkNum * 128 # 128sample/1chunk
        if unpackedData.shape[0] * 2 > convertSize:
            convertSize = unpackedData.shape[0] * 2

        # print("convert Size", convertChunkNum, convertSize)

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

