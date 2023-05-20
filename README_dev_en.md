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
$ cd voice-changer
```

## For Server Developer

1. Clone support VC repository

```
cd server
git clone https://github.com/isletennos/MMVC_Client.git MMVC_Client_v13
git clone https://github.com/isletennos/MMVC_Client.git MMVC_Client_v15
git clone https://github.com/StarStringStudio/so-vits-svc.git so-vits-svc-40
git clone https://github.com/StarStringStudio/so-vits-svc.git so-vits-svc-40v2
cd so-vits-svc-40v2 && git checkout 08c70ff3d2f7958820b715db2a2180f4b7f92f8d && cd -
git clone https://github.com/yxlllc/DDSP-SVC.git DDSP-SVC
git clone https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI.git RVC
```

2. Install requirements

```
$ pip install -r requirements.txt
```

3. Run server

Run server with the below command. You can replace the path to each weight.

```
$ python3 MMVCServerSIO.py -p 18888 --https true \
 --content_vec_500 pretrain/checkpoint_best_legacy_500.pt  \
 --hubert_base pretrain/hubert_base.pt \
 --hubert_soft pretrain/hubert/hubert-soft-0d54a1f4.pt \
 --nsf_hifigan pretrain/nsf_hifigan/model \
 --hubert_base_jp pretrain/rinna_hubert_base_jp.pt \
 --model_dir models
```

4. Enjoy developing.

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
