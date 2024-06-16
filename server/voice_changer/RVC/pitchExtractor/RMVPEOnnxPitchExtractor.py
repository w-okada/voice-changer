import numpy as np
from const import PitchExtractorType
from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.common.TorchUtils import circular_write
import onnxruntime
import torch
from onnxconverter_common import float16
import onnx
import os
from librosa.filters import mel
from librosa.util import pad_center
from scipy.signal import get_window
from typing import Optional
import torch.nn.functional as F


class RMVPEOnnxPitchExtractor(PitchExtractor):

    def __init__(self, file: str):
        super().__init__()
        self.file = file
        self.type: PitchExtractorType = "rmvpe_onnx"
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        device_manager = DeviceManager.get_instance()
        self.is_half = device_manager.use_fp16()
        (
            onnxProviders,
            onnxProviderOptions,
        ) = device_manager.get_onnx_execution_provider()

        if self.is_half:
            fname, _ = os.path.splitext(os.path.basename(file))
            fp16_fpath = os.path.join(os.path.dirname(file), f'{fname}.fp16.onnx')
            if not os.path.exists(fp16_fpath):
                model: onnx.ModelProto = float16.convert_float_to_float16(onnx.load(file))
                onnx.save(model, fp16_fpath)
            else:
                model = onnx.load(fp16_fpath)
        else:
            model = onnx.load(file)

        self.fp_dtype_t = torch.float16 if self.is_half else torch.float32
        self.fp_dtype_np = np.float16 if self.is_half else np.float32

        self.threshold = np.array(0.3, dtype=self.fp_dtype_np)

        so = onnxruntime.SessionOptions()
        # so.log_severity_level = 3
        # so.enable_profiling = True
        self.mel_extractor = MelSpectrogram(
            self.is_half, 128, 16000, 1024, 160, mel_fmin=30, mel_fmax=8000
        ).to(device_manager.device)
        self.onnx_session = onnxruntime.InferenceSession(model.SerializeToString(), sess_options=so, providers=onnxProviders, provider_options=onnxProviderOptions)

    def extract(self, audio: torch.Tensor, pitchf: torch.Tensor, f0_up_key: int, sr: int, window: int) -> tuple[torch.Tensor, torch.Tensor]:
        mel = self.mel_extractor(audio.unsqueeze(0).float())

        if audio.device.type == 'cuda':
            binding = self.onnx_session.io_binding()

            binding.bind_input('mel', device_type='cuda', device_id=audio.device.index, element_type=self.fp_dtype_np, shape=tuple(mel.shape), buffer_ptr=mel.data_ptr())
            binding.bind_cpu_input('threshold', self.threshold)

            binding.bind_output('pitchf', device_type='cuda', device_id=audio.device.index)

            self.onnx_session.run_with_iobinding(binding)

            output = [output.numpy() for output in binding.get_outputs()]
        else:
            output: list[np.ndarray] = self.onnx_session.run(
                ["pitchf"],
                {
                    "mel": mel.detach().cpu().numpy(),
                    "threshold": self.threshold,
                },
            )
        # self.onnx_session.end_profiling()

        f0 = torch.as_tensor(output[0], dtype=self.fp_dtype_t, device=audio.device).squeeze()

        f0 *= 2 ** (f0_up_key / 12)
        circular_write(f0, pitchf)
        f0_mel = 1127.0 * torch.log(1.0 + pitchf / 700.0)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = torch.round(f0_mel, out=f0_mel).to(dtype=torch.int64)
        return f0_coarse.unsqueeze(0), pitchf.unsqueeze(0)

# TODO: Refactor into separate modules
class STFT(torch.nn.Module):
    def __init__(
        self, filter_length=1024, hop_length=512, win_length=None, window="hann"
    ):
        """
        This module implements an STFT using 1D convolution and 1D transpose convolutions.
        This is a bit tricky so there are some cases that probably won't work as working
        out the same sizes before and after in all overlap add setups is tough. Right now,
        this code should work with hop lengths that are half the filter length (50% overlap
        between frames).

        Keyword Arguments:
            filter_length {int} -- Length of filters used (default: {1024})
            hop_length {int} -- Hop length of STFT (restrict to 50% overlap between frames) (default: {512})
            win_length {[type]} -- Length of the window function applied to each frame (if not specified, it
                equals the filter length). (default: {None})
            window {str} -- Type of window to use (options are bartlett, hann, hamming, blackman, blackmanharris)
                (default: {'hann'})
        """
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.window = window
        self.forward_transform = None
        self.pad_amount = int(self.filter_length / 2)
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )
        forward_basis = torch.FloatTensor(fourier_basis)
        inverse_basis = torch.FloatTensor(np.linalg.pinv(fourier_basis))

        assert filter_length >= self.win_length
        # get window and zero center pad it to filter_length
        fft_window = get_window(window, self.win_length, fftbins=True)
        fft_window = pad_center(fft_window, size=filter_length)
        fft_window = torch.from_numpy(fft_window).float()

        # window the bases
        forward_basis *= fft_window
        inverse_basis = (inverse_basis.T * fft_window).T

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())
        self.register_buffer("fft_window", fft_window.float())

    def transform(self, input_data, return_phase=False):
        """Take input data (audio) to STFT domain.

        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)

        Returns:
            magnitude {tensor} -- Magnitude of STFT with shape (num_batch,
                num_frequencies, num_frames)
            phase {tensor} -- Phase of STFT with shape (num_batch,
                num_frequencies, num_frames)
        """
        input_data = F.pad(
            input_data,
            (self.pad_amount, self.pad_amount),
            mode="reflect",
        )
        forward_transform = input_data.unfold(
            1, self.filter_length, self.hop_length
        ).permute(0, 2, 1)
        forward_transform = torch.matmul(self.forward_basis, forward_transform)
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        if return_phase:
            phase = torch.atan2(imag_part.data, real_part.data)
            return magnitude, phase
        else:
            return magnitude

    def inverse(self, magnitude, phase):
        """Call the inverse STFT (iSTFT), given magnitude and phase tensors produced
        by the ```transform``` function.

        Arguments:
            magnitude {tensor} -- Magnitude of STFT with shape (num_batch,
                num_frequencies, num_frames)
            phase {tensor} -- Phase of STFT with shape (num_batch,
                num_frequencies, num_frames)

        Returns:
            inverse_transform {tensor} -- Reconstructed audio given magnitude and phase. Of
                shape (num_batch, num_samples)
        """
        cat = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )
        fold = torch.nn.Fold(
            output_size=(1, (cat.size(-1) - 1) * self.hop_length + self.filter_length),
            kernel_size=(1, self.filter_length),
            stride=(1, self.hop_length),
        )
        inverse_transform = torch.matmul(self.inverse_basis, cat)
        inverse_transform = fold(inverse_transform)[
            :, 0, 0, self.pad_amount : -self.pad_amount
        ]
        window_square_sum = (
            self.fft_window.pow(2).repeat(cat.size(-1), 1).T.unsqueeze(0)
        )
        window_square_sum = fold(window_square_sum)[
            :, 0, 0, self.pad_amount : -self.pad_amount
        ]
        inverse_transform /= window_square_sum
        return inverse_transform

    def forward(self, input_data):
        """Take input data (audio) to STFT domain and then back to audio.

        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)

        Returns:
            reconstruction {tensor} -- Reconstructed audio given magnitude and phase. Of
                shape (num_batch, num_samples)
        """
        self.magnitude, self.phase = self.transform(input_data, return_phase=True)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class MelSpectrogram(torch.nn.Module):
    def __init__(
            self,
            is_half: bool,
            n_mel_channels: int,
            sampling_rate: int,
            win_length: int,
            hop_length: int,
            n_fft: Optional[int] = None,
            mel_fmin: int = 0,
            mel_fmax: Optional[int] = None,
            clamp: float = 1e-5,
    ):
        super(MelSpectrogram, self).__init__()
        n_fft = win_length if n_fft is None else n_fft
        mel_basis = mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp
        self.stft = STFT(
            filter_length=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window="hann",
        )
        self.is_half = is_half

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        magnitude = self.stft.transform(audio)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        if self.is_half:
            mel_output = mel_output.half()
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec
