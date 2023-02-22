

from features import SignalGenerator, dilated_factor
from scipy.interpolate import interp1d
import torch
import numpy as np
import json
import os
hann_window = {}


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """

    def __init__(
        self,
        sample_rate,
        hop_size,
        f0_factor=1.0,
        dense_factors=[0.5, 1, 4, 8],
        upsample_scales=[8, 4, 2, 2],
        sine_amp=0.1,
        noise_amp=0.003,
        signal_types=["sine"],
    ):
        self.dense_factors = dense_factors
        self.prod_upsample_scales = np.cumprod(upsample_scales)
        self.sample_rate = sample_rate
        self.signal_generator = SignalGenerator(
            sample_rate=sample_rate,
            hop_size=hop_size,
            sine_amp=sine_amp,
            noise_amp=noise_amp,
            signal_types=signal_types,
        )
        self.f0_factor = f0_factor

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid, note]
        """

        spec_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))
        spec_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), batch[0][0].size(1))
        f0_padded = torch.FloatTensor(len(batch), 1, batch[0][2].size(0))
        # 返り値の初期化
        spec_padded.zero_()
        f0_padded.zero_()

        # dfs
        dfs_batch = [[] for _ in range(len(self.dense_factors))]

        # row spec, sid, f0
        for i in range(len(batch)):
            row = batch[i]

            spec = row[0]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            sid[i] = row[1]
            # 推論時 f0/cf0にf0の倍率を乗算してf0/cf0を求める
            f0 = row[2] * self.f0_factor
            f0_padded[i, :, :f0.size(0)] = f0

            # dfs
            dfs = []
            # dilated_factor の入力はnumpy!!
            for df, us in zip(self.dense_factors, self.prod_upsample_scales):
                dfs += [
                    np.repeat(dilated_factor(torch.unsqueeze(f0, dim=1).to('cpu').detach().numpy(), self.sample_rate, df), us)
                ]

            # よくわからないけど、後で論文ちゃんと読む
            for i in range(len(self.dense_factors)):
                dfs_batch[i] += [
                    dfs[i].astype(np.float32).reshape(-1, 1)
                ]  # [(T', 1), ...]
        # よくわからないdfsを転置
        for i in range(len(self.dense_factors)):
            dfs_batch[i] = torch.FloatTensor(np.array(dfs_batch[i])).transpose(
                2, 1
            )  # (B, 1, T')

        # f0/cf0を実際に使うSignalに変換する
        in_batch = self.signal_generator(f0_padded)

        return spec_padded, spec_lengths, sid, in_batch, dfs_batch


def convert_continuos_f0(f0, f0_size):
    # get start and end of f0
    if (f0 == 0).all():
        return np.zeros((f0_size,))
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]
    # padding start and end of f0 sequence
    cf0 = f0
    start_idx = np.where(cf0 == start_f0)[0][0]
    end_idx = np.where(cf0 == end_f0)[0][-1]
    cf0[:start_idx] = start_f0
    cf0[end_idx:] = end_f0
    # get non-zero frame index
    nz_frames = np.where(cf0 != 0)[0]
    # perform linear interpolation
    f = interp1d(nz_frames, cf0[nz_frames], bounds_error=False, fill_value=0.0)
    cf0_ = f(np.arange(0, f0_size))
    # print(cf0.shape, cf0_.shape, f0.shape, f0_size)
    # print(cf0_)
    return f(np.arange(0, f0_size))


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def get_hparams_from_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path), f"No such file or directory: {checkpoint_path}"
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = {
        **checkpoint_dict['pe'],
        **checkpoint_dict['flow'],
        **checkpoint_dict['text_enc'],
        **checkpoint_dict['dec'],
        **checkpoint_dict['emb_g']
    }
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    return model, optimizer, learning_rate, iteration
