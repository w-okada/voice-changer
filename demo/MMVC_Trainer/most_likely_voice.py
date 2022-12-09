import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
import sys
import argparse

import utils
from data_utils import (
  TextAudioSpeakerLoader,
  TextAudioSpeakerCollate
)
from models import (
  SynthesizerTrn
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

def mel_loss(spec, audio, hps):
    # 学習と同じやり方でmel spectrogramの誤差を算出
    y_mel = spec_to_mel_torch(
        spec, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate,
        hps.data.mel_fmin, 
        hps.data.mel_fmax)

    y_hat = audio.unsqueeze(0).unsqueeze(0)
    y_hat = y_hat.float()
    y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1), 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate, 
        hps.data.hop_length, 
        hps.data.win_length, 
        hps.data.mel_fmin, 
        hps.data.mel_fmax)

    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
    return loss_mel

def run_most_likely_voice():
    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fine_model_path')
    parser.add_argument('-v', '--myvoice_path', default='./dataset/textful/00_myvoice/wav')
    parser.add_argument('-c', '--config_path', default='./configs/baseconfig.json')
    parser.add_argument('-n', '--sample_voice_num', default= 5)
    args = parser.parse_args()

    #load config
    hps = utils.get_hparams_from_file(args.config_path)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    #_ = net_g.eval()
    _ = utils.load_checkpoint(args.fine_model_path, net_g, None)

    dummy_source_speaker_id = 109
    #モデルに入れるための加工を行うためにTextAudioSpeakerLoaderを呼び出す
    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data, augmentation=False, no_use_textfile = True)
    wav_files = sorted(glob.glob(f"{args.myvoice_path}/*.wav"))
    wav_files = wav_files[:args.sample_voice_num]
    all_data = list()
    for wav_file in tqdm(wav_files):
        data = eval_dataset.get_audio_text_speaker_pair([wav_file, dummy_source_speaker_id, "a"])
        data = TextAudioSpeakerCollate()([data])
        all_data.append(data)

    speaker_num = 100
    loss_mels = np.zeros(speaker_num)

    for target_id in tqdm(range(0, speaker_num)):
        sid_target = torch.LongTensor([target_id]).cuda()
        print(f"target id: {target_id} / loss mel: ", end="")
        for data in tqdm(all_data):
            x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [x.cuda() for x in data]
            result = net_g.cuda().voice_conversion(spec, spec_lengths, sid_src=sid_target, sid_tgt=sid_target)
            audio = result[0][0,0]
            loss_mel = mel_loss(spec, audio, hps).data.cpu().float().numpy()
            loss_mels[target_id] += loss_mel
            print(f"{loss_mel:.3f} ", end="")
        loss_mels[target_id] /= len(all_data)
        print(f"/ ave: {loss_mels[target_id]:.3f}")

    print("--- Most likely voice ---")
    top_losses = np.argsort(loss_mels)[:3]
    for target_id in top_losses:
        print(f"target id: {target_id} / ave: {loss_mels[target_id]:.3f}")

if __name__ == "__main__":
    run_most_likely_voice()