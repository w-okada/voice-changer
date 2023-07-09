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
```

## サーバ開発者向け

1. モジュールをインストールする

```
$ cd voice-changer/server
$ pip install -r requirements.txt
```

2. サーバを起動する

次のコマンドで起動します。各種重みについてのパスは環境に合わせて変えてください。

```
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
    --model_dir model_dir \
    --samples samples.json

```

2-1. トラブルシュート

(1) OSError: PortAudio library not found
次のようなメッセージが表示される場合、追加でライブラリを追加する必要があります。

```
OSError: PortAudio library not found
```

ubuntu(wsl2)の場合下記のコマンドでインストールできます。

```
$ sudo apt-get install libportaudio2
$ sudo apt-get install libasound-dev
```

3. 開発しましょう

### Appendix

1. Win + Anaconda のとき (not supported)

pytorch を conda で入れないと gpu を認識しないかもしれない。

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

また、追加で下記も必要のようだ。

```
pip install chardet
pip install numpy==1.24.0
```

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
