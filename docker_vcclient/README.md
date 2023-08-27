## VC Client for Docker

[English](./README_en.md)

## ビルド

リポジトリフォルダのルートで

```
npm run build:docker:vcclient
```

## 実行

リポジトリフォルダのルートで

```
bash start_docker.sh
```

ブラウザ(Chrome のみサポート)でアクセスすると画面が表示されます。

## RUN with options

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
