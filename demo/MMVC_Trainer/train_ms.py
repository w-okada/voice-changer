import os
import json
import argparse
import itertools
import math
from psutil import cpu_count
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.cuda.amp import autocast, GradScaler
import datetime
import pytz
import time
from tqdm import tqdm
#import warnings


import commons
import utils
from data_utils import (
  TextAudioSpeakerLoader,
  TextAudioSpeakerCollate,
  DistributedBucketSampler
)
from models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

#stftの警告対策
#warnings.resetwarnings()
#warnings.simplefilter('ignore', UserWarning)
#warnings.simplefilter('ignore', DeprecationWarning)

torch.backends.cudnn.benchmark = True
global_step = 0


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '8000'

  hps = utils.get_hparams()

  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
  global global_step
  
  if hps.others.os_type == "windows":
    backend_type = "gloo"
    parallel = DP
  else: # Colab
    backend_type = "nccl"
    parallel = DDP

  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  cpu_count = os.cpu_count()
  if cpu_count > 8:
    cpu_count = 8

  dist.init_process_group(backend=backend_type, init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)
  train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data, augmentation=hps.augmentation.enable, augmentation_params=hps.augmentation)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [96,375,750,1125,1500,1875,2250,2625,3000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioSpeakerCollate()
  train_loader = DataLoader(train_dataset, num_workers=cpu_count, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)
  if rank == 0:
    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data, augmentation=False)
    eval_sampler = DistributedBucketSampler(
      eval_dataset,
      hps.train.batch_size,
      [96,375,750,1125,1500,1875,2250,2625,3000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
    eval_loader = DataLoader(eval_dataset, num_workers=cpu_count, shuffle=False, pin_memory=True,
        collate_fn=collate_fn, batch_sampler=eval_sampler)
  net_g = SynthesizerTrn(
      len(symbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      n_speakers=hps.data.n_speakers,
      **hps.model).cuda(rank)
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  net_g = parallel(net_g, device_ids=[rank])
  net_d = parallel(net_d, device_ids=[rank])

  logger.info('FineTuning : '+str(hps.fine_flag))
  if hps.fine_flag:
      logger.info('Load model : '+str(hps.fine_model_g))
      logger.info('Load model : '+str(hps.fine_model_d))
      _, _, _, epoch_str = utils.load_checkpoint(hps.fine_model_g, net_g, optim_g)
      _, _, _, epoch_str = utils.load_checkpoint(hps.fine_model_d, net_d, optim_d)
      epoch_str = 1
      global_step = 0

  else:
    try:
      _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
      _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
      global_step = (epoch_str - 1) * len(train_loader)
    except:
      epoch_str = 1
      global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler(enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None)
    scheduler_g.step()
    scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step

  net_g.train()
  net_d.train()
  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(tqdm(train_loader, desc="Epoch {}".format(epoch))):
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
    speakers = speakers.cuda(rank, non_blocking=True)

    with autocast(enabled=hps.train.fp16_run):
      y_hat, attn, ids_slice, x_mask, z_mask,\
      (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths, speakers)
      mel = spec_to_mel_torch(
          spec, 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate,
          hps.data.mel_fmin, 
          hps.data.mel_fmax)
      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
    y_hat = y_hat.float()
    y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1), 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate, 
        hps.data.hop_length, 
        hps.data.win_length, 
        hps.data.mel_fmin, 
        hps.data.mel_fmax
    )
    y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 

    # Discriminator
    y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
    with autocast(enabled=False):
      loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
      loss_disc_all = loss_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with autocast(enabled=hps.train.fp16_run):
      # Generator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
      with autocast(enabled=False):
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info(datetime.datetime.now(pytz.timezone('Asia/Tokyo')))
        logger.info([x.item() for x in losses] + [global_step, lr])
        
        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl})

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
        image_dict = { 
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
        }
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_latest_99999999.pth"))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_latest_99999999.pth"))

      if global_step % hps.train.eval_interval == 0 and global_step != 0:
        evaluate(hps, net_g, eval_loader, writer_eval, logger)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_latest_99999999.pth"))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_latest_99999999.pth"))
    global_step += 1

 
def evaluate(hps, generator, eval_loader, writer_eval, logger):
    scalar_dict = {}
    scalar_dict.update({"loss/g/mel": 0.0, "loss/g/kl": 0.0})
    with torch.no_grad():
      #evalのデータセットを一周する
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(tqdm(eval_loader, desc="Epoch {}".format("eval"))):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)
        speakers = speakers.cuda(0)
        #autocastはfp16のおまじない
        with autocast(enabled=hps.train.fp16_run):
          #Generator
          y_hat, attn, ids_slice, x_mask, z_mask,\
          (z, z_p, m_p, logs_p, m_q, logs_q) = generator(x, x_lengths, spec, spec_lengths, speakers)

          mel = spec_to_mel_torch(
              spec, 
              hps.data.filter_length, 
              hps.data.n_mel_channels, 
              hps.data.sampling_rate,
              hps.data.mel_fmin, 
              hps.data.mel_fmax)
          y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
        y_hat = y_hat.float()
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1), 
            hps.data.filter_length, 
            hps.data.n_mel_channels, 
            hps.data.sampling_rate, 
            hps.data.hop_length, 
            hps.data.win_length, 
            hps.data.mel_fmin, 
            hps.data.mel_fmax
        )
        batch_num = batch_idx

        y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 

        with autocast(enabled=hps.train.fp16_run):
          with autocast(enabled=False):
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

        scalar_dict["loss/g/mel"] = scalar_dict["loss/g/mel"] + loss_mel
        scalar_dict["loss/g/kl"] = scalar_dict["loss/g/kl"] + loss_kl
      
      #lossをepoch1周の結果をiter単位の平均値に
      scalar_dict["loss/g/mel"] = scalar_dict["loss/g/mel"] / (batch_num+1)
      scalar_dict["loss/g/kl"] = scalar_dict["loss/g/kl"] / (batch_num+1)
      logger.info("loss/g/mel : {} lloss/g/kl : {}".format(str(scalar_dict["loss/g/mel"]), str(scalar_dict["loss/g/kl"])))

    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      scalars=scalar_dict,
    )
                           
if __name__ == "__main__":
  main()
