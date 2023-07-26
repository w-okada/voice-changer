import glob
import logging
import os
import shutil
import socket
import sys

import ffmpeg
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import torch
from scipy.io.wavfile import read
from torch.nn import functional as F

from modules.shared import ROOT_DIR

from .config import TrainConfig

matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging


class AWP:
    """
    Fast AWP
    https://www.kaggle.com/code/junkoda/fast-awp
    """
    def __init__(self, model, optimizer, *, adv_param='weight',
                 adv_lr=0.01, adv_eps=0.01):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}

    def perturb(self):
        """
        Perturb model parameters for AWP gradient
        Call before loss and loss.backward()
        """
        self._save()  # save model parameters
        self._attack_step()  # perturb weights

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                grad = self.optimizer.state[param]['exp_avg']
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.detach())

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Set lower and upper limit in change
                    limit_eps = self.adv_eps * param.detach().abs()
                    param_min = param.data - limit_eps
                    param_max = param.data + limit_eps

                    # Perturb along gradient
                    # w += (adv_lr * |w| / |grad|) * grad
                    param.data.add_(grad, alpha=(self.adv_lr * (norm_data + e) / (norm_grad + e)))

                    # Apply the limit to the change
                    param.data.clamp_(param_min, param_max)

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.clone().detach()
                else:
                    self.backup[name].copy_(param.data)

    def restore(self):
        """
        Restore model parameter to correct position; AWP do not perturbe weights, it perturb gradients
        Call after loss.backward(), before optimizer.step()
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])


def load_audio(file: str, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # Prevent small white copy path head and tail with spaces and " and return
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()


def find_empty_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():  # 模型需要的shape
        try:
            new_state_dict[k] = saved_state_dict[k]
            if saved_state_dict[k].shape != state_dict[k].shape:
                print(
                    f"shape-{k}-mismatch|need-{state_dict[k].shape}|get-{saved_state_dict[k].shape}"
                )
                if saved_state_dict[k].dim() == 2:  # NOTE: check is this ok?
                    # for embedded input 256 <==> 768
                    # this achieves we can continue training from original's pretrained checkpoints when using embedder that 768-th dim output etc.
                    if saved_state_dict[k].dtype == torch.half:
                        new_state_dict[k] = (
                            F.interpolate(
                                saved_state_dict[k].float().unsqueeze(0).unsqueeze(0),
                                size=state_dict[k].shape,
                                mode="bilinear",
                            )
                            .half()
                            .squeeze(0)
                            .squeeze(0)
                        )
                    else:
                        new_state_dict[k] = (
                            F.interpolate(
                                saved_state_dict[k].unsqueeze(0).unsqueeze(0),
                                size=state_dict[k].shape,
                                mode="bilinear",
                            )
                            .squeeze(0)
                            .squeeze(0)
                        )
                    print(
                        "interpolated new_state_dict",
                        k,
                        "from",
                        saved_state_dict[k].shape,
                        "to",
                        new_state_dict[k].shape,
                    )
                else:
                    raise KeyError
        except Exception as e:
            # print(traceback.format_exc())
            print(f"{k} is not in the checkpoint")
            print("error: %s" % e)
            new_state_dict[k] = v  # 模型自带的随机值
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    print("Loaded model weights")

    epoch = checkpoint_dict["epoch"]
    learning_rate = checkpoint_dict["learning_rate"]
    if optimizer is not None and load_opt == 1:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, epoch))
    return model, optimizer, learning_rate, epoch


def save_state(model, optimizer, learning_rate, epoch, checkpoint_path):
    print(
        "Saving model and optimizer state at epoch {} to {}".format(
            epoch, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    filelist = glob.glob(os.path.join(dir_path, regex))
    if len(filelist) == 0:
        return None
    filelist.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    filepath = filelist[-1]
    return filepath


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
    )
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_config(training_dir: str, sample_rate: int, emb_channels: int):
    if emb_channels == 256:
        config_path = os.path.join(ROOT_DIR, "configs", f"{sample_rate}.json")
    else:
        config_path = os.path.join(
            ROOT_DIR, "configs", f"{sample_rate}-{emb_channels}.json"
        )
    config_save_path = os.path.join(training_dir, "config.json")

    shutil.copyfile(config_path, config_save_path)

    return TrainConfig.parse_file(config_save_path)
