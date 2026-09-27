## For Developper

[Japanese](/README_dev_ja.md) [Russian](/README_dev_ru.md)

## Prerequisit

- Linux(ubuntu, debian) or WSL2, (not tested for other linux distributions and Mac)
- Anaconda

## Preparation

1. Create anaconda virtual environment

```
$ conda create -n vcclient-dev python=3.10
$ conda activate vcclient-dev
```

2. clone repository

```
$ git clone https://github.com/w-okada/voice-changer.git
$ cd voice-changer
```

## For Server Developer

1. Install requirements

1-1. For No GPU

```
$ python -m pip install -r server/requirements_cpuonly.txt
```

1-2. For NVIDIA GPUs

Please rewrite the `cu128` part in `server/requirements_nvidia.txt` according to your CUDA version.
The default `cu128` is for CUDA 12.8.

```
--extra-index-url https://download.pytorch.org/whl/cu128

torch==2.7.0+cu128
torchaudio==2.7.0+cu128
```

Next, execute the following:

```
$ python -m pip install -r server/requirements_nvidia.txt
```

2. Run server

Run server with the below command. You can replace the path to each weight.

```
$ cd server
$ python3 MMVCServerSIO.py -p 18888 --https true \
    --content_vec_500 pretrain/checkpoint_best_legacy_500.pt  \
    --content_vec_500_onnx pretrain/content_vec_500.onnx \
    --content_vec_500_onnx_on true \
    --hubert_base pretrain/hubert_base.pt \
    --hubert_base_jp pretrain/rinna_hubert_base_jp.pt \
    --hubert_soft pretrain/hubert/hubert-soft-0d54a1f4.pt \
    --nsf_hifigan pretrain/nsf_hifigan/model \
    --crepe_onnx_full pretrain/crepe_onnx_full.onnx \
    --crepe_onnx_tiny pretrain/crepe_onnx_tiny.onnx \
    --rmvpe pretrain/rmvpe.pt \
    --model_dir model_dir \
    --samples samples.json

```

Access with Browser (currently only chrome is supported), then you can see gui.

2-1. Trouble shoot

(1) OSError: PortAudio library not found
If you get the message below, you shold install additional library.

```
OSError: PortAudio library not found
```

You can install the library this command.

```
$ sudo apt-get install libportaudio2
$ sudo apt-get install libasound-dev
```

(2) It's not starting up! Damn software!

The client will not start automatically. Please launch your browser and access the URL displayed on the console. And watch your words.

(3) Could not load library libcudnn_cnn_infer.so.8

When using WSL, you might encounter a message saying `Could not load library libcudnn_cnn_infer.so.8. Error: libcuda.so: cannot open shared object file: No such file or directory`. This often happens because the path hasn't been properly set. Please set the path as shown below. It might be handy to add this to your launch script, such as .bashrc.

```
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

- reference
  - https://qiita.com/cacaoMath/items/811146342946cdde5b83
  - https://github.com/microsoft/WSL/issues/8587

3. Enjoy developing.

### Appendix

1. Win + Anaconda (not supported)

use conda to install pytorch

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Also run these command.

```
pip install chardet
pip install numpy==1.24.0
```

## For Client Developer

1. Import modules and initial build

```
cd client
cd lib
npm install
npm run build:dev
cd ../demo
npm install
npm run build:dev
```

2. Enjoy developing.
