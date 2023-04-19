## VC Client for Docker

[English](/README_en.md)

## ビルド

リポジトリフォルダのルートで

```
npm run build:docker:vcclient
```

## 実行準備

`docker_vcclient/weights`に外部のモデルの重みを配置する。

配置後のイメージは次のような感じです。配置する重みは使用する VC の種類によって異なります。

```
$ tree docker_vcclient/weights/
docker_vcclient/weights/
├── checkpoint_best_legacy_500.onnx
├── checkpoint_best_legacy_500.pt
├── hubert-soft-0d54a1f4.pt
├── hubert_base.pt
└── nsf_hifigan
    ├── NOTICE.txt
    ├── NOTICE.zh-CN.txt
    ├── config.json
    └── model
```

## 実行

リポジトリフォルダのルートで

```
bash start_docker.sh
```

GPU を使用しない場合は

```
USE_GPU=off bash start_docker.sh
```

ポート番号を変えたい場合は

```
EX_PORT=<port> bash start_docker.sh
```

ローカルのイメージを使用したい場合は

```
USE_LOCAL=on bash start_docker.sh
```

## Push to Repo (only for devs)

```
npm run push:docker:vcclient
```
