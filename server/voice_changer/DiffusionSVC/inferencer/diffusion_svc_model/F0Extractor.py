
from torchaudio.transforms import Resample
import pyworld as pw
import numpy as np
import torchcrepe
import torch
import torch.nn.functional as F
CREPE_RESAMPLE_KERNEL = {}


def median_pool_1d(x, kernel_size):
    x = x.unsqueeze(1)
    x = F.pad(x, ((kernel_size - 1) // 2, kernel_size // 2), mode="reflect")
    x = x.squeeze(1)
    x = x.unfold(1, kernel_size, 1)
    x, _ = torch.sort(x, dim=-1)
    return x[:, :, (kernel_size - 1) // 2]


def masked_avg_pool_1d(x, kernel_size):
    x = x.unsqueeze(1)
    x = F.pad(x, ((kernel_size - 1) // 2, kernel_size // 2), mode="reflect")
    mask = ~torch.isnan(x)
    masked_x = torch.where(mask, x, torch.zeros_like(x))
    ones_kernel = torch.ones(x.size(1), 1, kernel_size, device=x.device)

    # Perform sum pooling
    sum_pooled = F.conv1d(
        masked_x,
        ones_kernel,
        stride=1,
        padding=0,
        groups=x.size(1),
    )

    # Count the non-masked (valid) elements in each pooling window
    valid_count = F.conv1d(
        mask.float(),
        ones_kernel,
        stride=1,
        padding=0,
        groups=x.size(1),
    )
    valid_count = valid_count.clamp(min=1)  # Avoid division by zero

    # Perform masked average pooling
    avg_pooled = sum_pooled / valid_count

    return avg_pooled.squeeze(1)


class F0_Extractor:
    def __init__(self, f0_extractor, sample_rate=44100, hop_size=512, f0_min=65, f0_max=800,
                 block_size=None, model_sampling_rate=None):
        self.block_size = block_size
        self.model_sampling_rate = model_sampling_rate
        self.f0_extractor = f0_extractor
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.transformer_f0 = None
        if f0_extractor == 'crepe':
            key_str = str(sample_rate)
            if key_str not in CREPE_RESAMPLE_KERNEL:
                CREPE_RESAMPLE_KERNEL[key_str] = Resample(sample_rate, 16000, lowpass_filter_width=128)
            self.resample_kernel = CREPE_RESAMPLE_KERNEL[key_str]
        if (self.block_size is not None) or (self.model_sampling_rate is not None):
            assert (self.block_size is not None) and (self.model_sampling_rate is not None)
            self.hop_size_follow_input = True
        else:
            self.hop_size_follow_input = False

    @torch.no_grad()
    def extract_f0(self, audio, key=0, sr=44100, silence_front=0):
        f0 = self.extract(audio.cpu().numpy(), uv_interp=True, silence_front=silence_front, sr=sr)
        # f0 = torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        print("[PITCH_F0_ORG1]", f0)
        f0 = torch.from_numpy(f0).float().unsqueeze(-1).unsqueeze(0)
        f0 = f0 * 2 ** (float(key) / 12)
        print("[PITCH_F0_ORG2]", f0)
        return f0

    def extract(self, audio, uv_interp=False, device=None, silence_front=0, sr=None):  # audio: 1d numpy array
        if sr is not None:
            assert self.hop_size_follow_input
            self.hop_size = self.block_size * sr / self.model_sampling_rate
            if (self.f0_extractor == 'crepe') and (sr != self.sample_rate):
                key_str = str(sr)
                if key_str not in CREPE_RESAMPLE_KERNEL:
                    CREPE_RESAMPLE_KERNEL[key_str] = Resample(sr, 16000, lowpass_filter_width=128)
                self.resample_kernel = CREPE_RESAMPLE_KERNEL[key_str]
            self.sample_rate = sr

        # extractor start time
        raw_audio = audio
        n_frames = int(len(audio) // self.hop_size) + 1

        start_frame = int(silence_front * self.sample_rate / self.hop_size)
        real_silence_front = start_frame * self.hop_size / self.sample_rate
        audio = audio[int(np.round(real_silence_front * self.sample_rate)):]

        if self.f0_extractor == 'dio':
            _f0, t = pw.dio(
                audio.astype('double'),
                self.sample_rate,
                f0_floor=self.f0_min,
                f0_ceil=self.f0_max,
                channels_in_octave=2,
                frame_period=(1000 * self.hop_size / self.sample_rate))
            f0 = pw.stonemask(audio.astype('double'), _f0, t, self.sample_rate)
            f0 = np.pad(f0.astype('float'), (start_frame, n_frames - len(f0) - start_frame))

        # extract f0 using harvest
        elif self.f0_extractor == 'harvest':
            print("[SRC AUDIO2]", audio[:10])
            print("_____hopsize______", (1000 * self.hop_size / self.sample_rate), self.sample_rate)
            f0, _ = pw.harvest(
                audio.astype('double'),
                self.sample_rate,
                f0_floor=self.f0_min,
                f0_ceil=self.f0_max,
                frame_period=(1000 * self.hop_size / self.sample_rate))
            print("[HARVEST-----1111]", f0)
            f0 = np.pad(f0.astype('float'), (start_frame, n_frames - len(f0) - start_frame))
            print("[HARVEST-----1112]", f0)

        # extract f0 using crepe
        elif self.f0_extractor == 'crepe':
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            resample_kernel = self.resample_kernel.to(device)
            wav16k_torch = resample_kernel(torch.FloatTensor(audio).unsqueeze(0).to(device))

            f0, pd = torchcrepe.predict(wav16k_torch, 16000, 80, self.f0_min, self.f0_max, pad=True, model='full',
                                        batch_size=512, device=device, return_periodicity=True)
            pd = median_pool_1d(pd, 4)
            f0 = torchcrepe.threshold.At(0.05)(f0, pd)
            f0 = masked_avg_pool_1d(f0, 4)

            f0 = f0.squeeze(0).cpu().numpy()
            f0 = np.array(
                [f0[int(min(int(np.round(n * self.hop_size / self.sample_rate / 0.005)), len(f0) - 1))] for n in
                 range(n_frames - start_frame)])
            f0 = np.pad(f0, (start_frame, 0))

        elif self.f0_extractor == "transformer_f0":
            if self.transformer_f0 is None:
                from transformer_f0.model import TransformerF0Infer
                self.transformer_f0 = TransformerF0Infer(model_path='exp/f0_test_genshin/model_540000.pt')
            # raw_audio = audio
            f0 = self.transformer_f0(audio=raw_audio, sr=self.sample_rate)
            # f0 = f0.transpose(1, 2)
            # f0 = torch.nn.functional.interpolate(f0, size=int(n_frames), mode='nearest')
            # f0 = f0.transpose(1, 2)
            f0 = f0.squeeze().cpu().numpy()
            # f0 = np.pad(f0.astype('float'), (start_frame, n_frames - len(f0) - start_frame))
        else:
            raise ValueError(f" [x] Unknown f0 extractor: {self.f0_extractor}")

        # interpolate the unvoiced f0
        if uv_interp:
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            f0[f0 < self.f0_min] = self.f0_min

        print("[HARVEST-----1113]", f0)
        return f0
