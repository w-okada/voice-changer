import numpy as np
import parselmouth
import torch
import pdb
from time import time as ttime
import torch.nn.functional as F
from config import x_pad, x_query, x_center, x_max
import scipy.signal as signal
import pyworld
import os
import traceback
import faiss
# from .const import RVC_MODEL_TYPE_NORMAL, RVC_MODEL_TYPE_PITCHLESS, RVC_MODEL_TYPE_WEBUI_256_NORMAL, RVC_MODEL_TYPE_WEBUI_768_NORMAL, RVC_MODEL_TYPE_WEBUI_256_PITCHLESS, RVC_MODEL_TYPE_WEBUI_768_PITCHLESS
from .const import RVC_MODEL_TYPE_RVC, RVC_MODEL_TYPE_WEBUI


class VC(object):
    def __init__(self, tgt_sr, device, is_half, x_pad):
        self.sr = 16000  # hubert输入采样率
        self.window = 160  # 每帧点数
        self.t_pad = self.sr * x_pad  # 每条前后pad时间
        self.t_pad_tgt = tgt_sr * x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * x_query  # 查询切点前后查询时间
        self.t_center = self.sr * x_center  # 查询切点位置
        self.t_max = self.sr * x_max  # 免查询时长阈值
        self.device = device
        self.is_half = is_half

    def get_f0(self, audio, p_len, f0_up_key, f0_method, inp_f0=None, silence_front=0):

        n_frames = int(len(audio) // self.window) + 1
        start_frame = int(silence_front * self.sr / self.window)
        real_silence_front = start_frame * self.window / self.sr

        audio = audio[int(np.round(real_silence_front * self.sr)):]

        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        if (f0_method == "pm"):
            f0 = parselmouth.Sound(audio, self.sr).to_pitch_ac(
                time_step=time_step / 1000, voicing_threshold=0.6,
                pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
            pad_size = (p_len - len(f0) + 1) // 2
            if (pad_size > 0 or p_len - len(f0) - pad_size > 0):
                f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode='constant')
        elif (f0_method == "harvest"):
            f0, t = pyworld.harvest(
                audio.astype(np.double),
                fs=self.sr,
                f0_ceil=f0_max,
                frame_period=10,
            )
            f0 = pyworld.stonemask(audio.astype(np.double), f0, t, self.sr)
            f0 = signal.medfilt(f0, 3)

            f0 = np.pad(f0.astype('float'), (start_frame, n_frames - len(f0) - start_frame))
        else:
            print("[Voice Changer] invalid f0 detector, use pm.", f0_method)
            f0 = parselmouth.Sound(audio, self.sr).to_pitch_ac(
                time_step=time_step / 1000, voicing_threshold=0.6,
                pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
            pad_size = (p_len - len(f0) + 1) // 2
            if (pad_size > 0 or p_len - len(f0) - pad_size > 0):
                f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode='constant')

        f0 *= pow(2, f0_up_key / 12)
        # with open("test.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        tf0 = self.sr // self.window  # 每秒f0点数
        if (inp_f0 is not None):
            delta_t = np.round((inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1).astype("int16")
            replace_f0 = np.interp(list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1])
            shape = f0[x_pad * tf0:x_pad * tf0 + len(replace_f0)].shape[0]
            f0[x_pad * tf0:x_pad * tf0 + len(replace_f0)] = replace_f0[:shape]
        # with open("test_opt.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int)
        return f0_coarse, f0bak  # 1-0

    def vc(self, model, net_g, sid, audio0, pitch, pitchf, times, index, big_npy, index_rate, f0=True, embChannels=256):  # ,file_index,file_big_npy
        feats = torch.from_numpy(audio0)
        if (self.is_half == True):
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
        if embChannels == 256:
            inputs = {
                "source": feats.to(self.device),
                "padding_mask": padding_mask,
                "output_layer": 9,  # layer 9
            }
        else:
            inputs = {
                "source": feats.to(self.device),
                "padding_mask": padding_mask,
            }

        t0 = ttime()
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            if embChannels == 256:
                feats = model.final_proj(logits[0])
            else:
                feats = logits[0]

        if (isinstance(index, type(None)) == False and isinstance(big_npy, type(None)) == False and index_rate != 0):
            npy = feats[0].cpu().numpy()
            if (self.is_half == True):
                npy = npy.astype("float32")
            D, I = index.search(npy, 1)
            npy = big_npy[I.squeeze()]
            if (self.is_half == True):
                npy = npy.astype("float16")
            feats = torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        t1 = ttime()
        p_len = audio0.shape[0] // self.window
        if (feats.shape[1] < p_len):
            p_len = feats.shape[1]
            if (pitch != None and pitchf != None):
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]
        p_len = torch.tensor([p_len], device=self.device).long()

        with torch.no_grad():
            if f0 == True:
                audio1 = (net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0] * 32768).data.cpu().float().numpy().astype(np.int16)
            else:
                if hasattr(net_g, "infer_pitchless"):
                    audio1 = (net_g.infer_pitchless(feats, p_len, sid)[0][0, 0] * 32768).data.cpu().float().numpy().astype(np.int16)
                else:
                    audio1 = (net_g.infer(feats, p_len, sid)[0][0, 0] * 32768).data.cpu().float().numpy().astype(np.int16)

            # audio1 = (net_g.infer(feats, p_len, None, pitchf, sid)[0][0, 0] * 32768).data.cpu().float().numpy().astype(np.int16)

        del feats, p_len, padding_mask
        torch.cuda.empty_cache()
        t2 = ttime()
        times[0] += (t1 - t0)
        times[2] += (t2 - t1)
        return audio1

    def pipeline(self, model, net_g, sid, audio, times, f0_up_key, f0_method, file_index, file_big_npy, index_rate, if_f0, f0_file=None, silence_front=0, f0=True, embChannels=256):
        if (file_big_npy != "" and file_index != "" and os.path.exists(file_big_npy) == True and os.path.exists(file_index) == True and index_rate != 0):
            try:
                index = faiss.read_index(file_index)
                big_npy = np.load(file_big_npy)
            except:
                traceback.print_exc()
                index = big_npy = None
        else:
            index = big_npy = None

        audio_opt = []
        t = None
        t1 = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode='reflect')
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None

        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if (if_f0 == 1):
            pitch, pitchf = self.get_f0(audio_pad, p_len, f0_up_key, f0_method, inp_f0, silence_front=silence_front)
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device, dtype=torch.float).unsqueeze(0)

        t2 = ttime()
        times[1] += (t2 - t1)
        if self.t_pad_tgt == 0:
            audio_opt.append(self.vc(model, net_g, sid, audio_pad[t:], pitch[:, t // self.window:]if t is not None else pitch,
                                     pitchf[:, t // self.window:]if t is not None else pitchf, times, index, big_npy, index_rate, f0, embChannels))
        else:
            audio_opt.append(self.vc(model, net_g, sid, audio_pad[t:], pitch[:, t // self.window:]if t is not None else pitch,
                                     pitchf[:, t // self.window:]if t is not None else pitchf, times, index, big_npy, index_rate, f0, embChannels)[self.t_pad_tgt:-self.t_pad_tgt])

        audio_opt = np.concatenate(audio_opt)
        del pitch, pitchf, sid
        torch.cuda.empty_cache()
        return audio_opt
