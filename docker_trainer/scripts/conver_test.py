import sys
sys.path.append(".")  # sifiganへのパスが必要。
import argparse

import torch

import numpy as np
from scipy.io.wavfile import write, read
import pyworld as pw
from logging import getLogger

# import utils
from models import SynthesizerTrn

# from mmvc_client import Hyperparameters # <- pyaudioなどが必要になるため必要なロジックのみコピペ
from client_modules import convert_continuos_f0, spectrogram_torch, TextAudioSpeakerCollate, get_hparams_from_file, load_checkpoint

logger = getLogger(__name__)


def setupArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, required=True, help="path for the config.json")
    parser.add_argument("-m", type=str, help="path for the pytorch model file")
    parser.add_argument("-o", type=str, help="path for the onnx model file")
    parser.add_argument("-s", type=int, required=True, help="source speaker id")
    parser.add_argument("-t", type=int, required=True, help="target speaker id")
    parser.add_argument("--input", type=str, required=True, help="input wav file")
    parser.add_argument("--output", type=str, required=True, help="outpu wav file")
    parser.add_argument("--f0_scale", type=float, required=True, help="f0 scale")
    return parser


def create_model(hps, pytorch_model_file):
    net_g = SynthesizerTrn(
        spec_channels=hps.data.filter_length // 2 + 1,
        segment_size=hps.train.segment_size // hps.data.hop_length,
        inter_channels=hps.model.inter_channels,
        hidden_channels=hps.model.hidden_channels,
        upsample_rates=hps.model.upsample_rates,
        upsample_initial_channel=hps.model.upsample_initial_channel,
        upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
        n_flow=hps.model.n_flow,
        dec_out_channels=1,
        dec_kernel_size=7,
        n_speakers=hps.data.n_speakers,
        gin_channels=hps.model.gin_channels,
        requires_grad_pe=hps.requires_grad.pe,
        requires_grad_flow=hps.requires_grad.flow,
        requires_grad_text_enc=hps.requires_grad.text_enc,
        requires_grad_dec=hps.requires_grad.dec
    )
    _ = net_g.eval()

    _ = load_checkpoint(pytorch_model_file, net_g, None)
    return net_g


def convert(hps, ssid, tsid, input, output, f0_scale):
    sr, signal = read(input)
    signal = signal / hps.data.max_wav_value
    _f0, _time = pw.dio(signal, hps.data.sampling_rate, frame_period=5.5)
    f0 = pw.stonemask(signal, _f0, _time, hps.data.sampling_rate)
    f0 = convert_continuos_f0(f0, int(signal.shape[0] / hps.data.hop_length))
    f0 = torch.from_numpy(f0.astype(np.float32))
    signal = torch.from_numpy(signal.astype(np.float32)).clone()
    signal = signal.unsqueeze(0)
    spec = spectrogram_torch(signal, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, center=False)
    spec = torch.squeeze(spec, 0)
    sid = torch.LongTensor([int(ssid)])
    data = TextAudioSpeakerCollate(
        sample_rate=hps.data.sampling_rate,
        hop_size=hps.data.hop_length,
        f0_factor=f0_scale
    )([(spec, sid, f0)])

    spec, spec_lengths, sid_src, sin, d = data
    spec = spec.cuda()
    spec_lengths = spec_lengths.cuda()
    sid_src = sid_src.cuda()
    sin = sin.cuda()
    d = tuple([d[:1].cuda() for d in d])
    sid_target = torch.LongTensor([tsid]).cuda()
    audio = net_g.cuda().voice_conversion(spec, spec_lengths, sin, d, sid_src, sid_target)[0, 0].data.cpu().float().numpy()
    # print(audio)
    write(output, 24000, audio)


if __name__ == '__main__':
    print("main")
    parser = setupArgParser()
    args = parser.parse_args()

    CONFIG_PATH = args.c
    hps = get_hparams_from_file(CONFIG_PATH)
    pytorch_model_file = args.m
    ssid = args.s
    tsid = args.t
    input = args.input
    output = args.output
    f0_scale = args.f0_scale

    net_g = create_model(hps, pytorch_model_file)
    convert(hps, ssid, tsid, input, output, f0_scale)
