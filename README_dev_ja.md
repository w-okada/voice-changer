## 開発者向け

[English](/README_dev_en.md)

## 前提

- Linux or WSL2 (not tested for Mac )
- Anaconda

## 準備

1. Anaconda の仮想環境を作成する

```
$ conda create -n vcclient-dev python=3.10
$ conda activate vcclient-dev
```

2. リポジトリをクローンする

```
$ git clone https://github.com/w-okada/voice-changer.git
$ cd voice-changer
```

## サーバ開発者向け

1. 外部のリポジトリをサーバ内にクローンする

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

2. モジュールをインストールする

```
$ pip install -r requirements.txt
```

3. サーバを起動する

次のコマンドで起動します。各種重みについてのパスは環境に合わせて変えてください。

```
$ python3 MMVCServerSIO.py -p 18888 --https true \
 --content_vec_500 pretrain/checkpoint_best_legacy_500.pt  \
 --hubert_base pretrain/hubert_base.pt \
 --hubert_soft pretrain/hubert/hubert-soft-0d54a1f4.pt \
 --nsf_hifigan pretrain/nsf_hifigan/model \
 --hubert_base_jp pretrain/rinna_hubert_base_jp.pt \
 --model_dir models
```

4. 開発しましょう

## クライアント開発者向け

1. モジュールをインストールして、一度ビルドします

```
cd client
cd lib
npm install
npm run build:dev
cd ../demo
npm install
npm run build:dev
```

2. 開発しましょう
