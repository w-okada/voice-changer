## For Developper

[Japanese](/README_dev_ja.md)

## Prerequisit

- Linux or WSL2 (not tested for Mac )
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
```

## For Server Developer
1. Install requirements

```
$ cd voice-changer/server
$ pip install -r requirements.txt
```

2. Run server

Run server with the below command. You can replace the path to each weight.

```
$ python3 MMVCServerSIO.py -p 18888 --https true \
 --content_vec_500 pretrain/checkpoint_best_legacy_500.pt  \
 --hubert_base pretrain/hubert_base.pt \
 --hubert_soft pretrain/hubert/hubert-soft-0d54a1f4.pt \
 --nsf_hifigan pretrain/nsf_hifigan/model \
 --hubert_base_jp pretrain/rinna_hubert_base_jp.pt \
 --model_dir model_dir
```

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

3. Enjoy developing.

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
