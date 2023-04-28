import numpy as np

# import parselmouth
import torch
import torch.nn.functional as F
import scipy.signal as signal
import pyworld


class VC(object):
    def __init__(self, tgt_sr, device, is_half, x_pad):
        self.sr = 16000  # hubert输入采样率
        self.window = 160  # 每帧点数
        self.t_pad = self.sr * x_pad  # 每条前后pad时间
        self.t_pad_tgt = tgt_sr * x_pad
        self.device = device
        self.is_half = is_half

    def get_f0(self, audio, p_len, f0_up_key, f0_method, silence_front=0):
        n_frames = int(len(audio) // self.window) + 1
        start_frame = int(silence_front * self.sr / self.window)
        real_silence_front = start_frame * self.window / self.sr

        silence_front_offset = int(np.round(real_silence_front * self.sr))
        audio = audio[silence_front_offset:]

        # time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        if f0_method == "dio":
            _f0, t = pyworld.dio(
                audio.astype(np.double),
                self.sr,
                f0_floor=f0_min,
                f0_ceil=f0_max,
                channels_in_octave=2,
                frame_period=10,
            )
            f0 = pyworld.stonemask(audio.astype(np.double), _f0, t, self.sr)
            f0 = np.pad(
                f0.astype("float"), (start_frame, n_frames - len(f0) - start_frame)
            )
        else:
            f0, t = pyworld.harvest(
                audio.astype(np.double),
                fs=self.sr,
                f0_ceil=f0_max,
                frame_period=10,
            )
            f0 = pyworld.stonemask(audio.astype(np.double), f0, t, self.sr)
            f0 = signal.medfilt(f0, 3)

            f0 = np.pad(
                f0.astype("float"), (start_frame, n_frames - len(f0) - start_frame)
            )

        f0 *= pow(2, f0_up_key / 12)
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int)

        # Volume Extract
        # volume = self.extractVolume(audio, 512)
        # volume = np.pad(
        #     volume.astype("float"), (start_frame, n_frames - len(volume) - start_frame)
        # )

        # return f0_coarse, f0bak, volume  # 1-0
        return f0_coarse, f0bak

    # def extractVolume(self, audio, hopsize):
    #     n_frames = int(len(audio) // hopsize) + 1
    #     audio2 = audio**2
    #     audio2 = np.pad(
    #         audio2,
    #         (int(hopsize // 2), int((hopsize + 1) // 2)),
    #         mode="reflect",
    #     )
    #     volume = np.array(
    #         [
    #             np.mean(audio2[int(n * hopsize) : int((n + 1) * hopsize)])  # noqa:E203
    #             for n in range(n_frames)
    #         ]
    #     )
    #     volume = np.sqrt(volume)
    #     return volume

    def pipeline(
        self,
        embedder,
        model,
        sid,
        audio,
        f0_up_key,
        f0_method,
        index,
        big_npy,
        index_rate,
        if_f0,
        silence_front=0,
        embChannels=256,
    ):
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()

        # ピッチ検出
        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(
                audio_pad,
                p_len,
                f0_up_key,
                f0_method,
                silence_front=silence_front,
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(
                pitchf, device=self.device, dtype=torch.float
            ).unsqueeze(0)

        # tensor
        feats = torch.from_numpy(audio_pad)
        if self.is_half is True:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)

        # embedding
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

        with torch.no_grad():
            logits = embedder.extract_features(**inputs)
            if embChannels == 256:
                feats = embedder.final_proj(logits[0])
            else:
                feats = logits[0]

        # Index - feature抽出
        if (
            isinstance(index, type(None)) is False
            and isinstance(big_npy, type(None)) is False
            and index_rate != 0
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half is True:
                npy = npy.astype("float32")
            D, I = index.search(npy, 1)
            npy = big_npy[I.squeeze()]
            if self.is_half is True:
                npy = npy.astype("float16")

            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        #
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        # ピッチ抽出
        p_len = audio_pad.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]
        p_len = torch.tensor([p_len], device=self.device).long()

        # 推論実行
        with torch.no_grad():
            if pitch is not None:
                audio1 = (
                    (model.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0] * 32768)
                    .data.cpu()
                    .float()
                    .numpy()
                    .astype(np.int16)
                )
            else:
                if hasattr(model, "infer_pitchless"):
                    audio1 = (
                        (model.infer_pitchless(feats, p_len, sid)[0][0, 0] * 32768)
                        .data.cpu()
                        .float()
                        .numpy()
                        .astype(np.int16)
                    )
                else:
                    audio1 = (
                        (model.infer(feats, p_len, sid)[0][0, 0] * 32768)
                        .data.cpu()
                        .float()
                        .numpy()
                        .astype(np.int16)
                    )

        del feats, p_len, padding_mask
        torch.cuda.empty_cache()

        if self.t_pad_tgt != 0:
            offset = self.t_pad_tgt
            end = -1 * self.t_pad_tgt
            audio1 = audio1[offset:end]

        del pitch, pitchf, sid
        torch.cuda.empty_cache()
        return audio1
