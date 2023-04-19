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

2. Copy weights

copy weights of emmbedding or vocoder. These file path can be set as parameter when invoke server.

(1) hubert_base.pt

download from [here](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main)

(2) content vec

download ContentVec_legacy_500 from [here](https://github.com/auspicious3000/contentvec)

(3) hubert_soft

download from [here](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)

(4) vocorder

download nsf_hifigan_20221211.zip from [here](https://github.com/openvpi/vocoders/releases/tag/nsf-hifigan-v1) and expand it.

3. Install requirements

```
$ pip install -r requirements.txt
```

4. Run server

Run server with the below command. You can replace the path to each weight.

```
$ python3 MMVCServerSIO.py -p 18888 --https true \
  --content_vec_500 weights/checkpoint_best_legacy_500.pt \
  --hubert_base weights/hubert_base.pt \
  --hubert_soft weights/hubert-soft-0d54a1f4.pt \
  --nsf_hifigan weights/nsf_hifigan/model
```

5. Enjoy developing.

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
