Voice Changer AI Trainer and Player Container
----

[MMVC](https://github.com/isletennos/MMVC_Trainer)のトレーニングと実行を簡単にするためのDockerコンテナです。

# 使用方法
## 前提
## Docker
Dockerを使えるようにしておいてください。
WindowsはWSL2上で使えるようにしておいてください。

トレーニング時にはGPUを見えるようにしておいてください。
```
$ docker run --gpus all --rm nvidia/cuda nvidia-smi
```
を実行して
```
Sun Sep 15 22:40:52 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 430.26       Driver Version: 430.26       CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 106...  Off  | 00000000:01:00.0  On |                  N/A |
| 38%   32C    P8     6W / 120W |      2MiB /  3016MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
こんな感じの出力が出ればOKです。

ボイスチェンジ時にはGPUは必須ではありません。あればより高速にボイスチェンジできるかもしれません。

## トレーニングデータの準備
### Data作成（自分の声）

[MMVC](https://github.com/isletennos/MMVC_Trainer)のドキュメントに従って、ITAコーパスなどの台本データ(textデータ)を入手し、でwavデータを作成してください。textデータとwavデータは、dataset/00_myvoiceフォルダの下においてください。

wavデータの作成は、こちらの[voice-recorder](https://github.com/w-okada/voice-recorder)アプリケーションを使用しても作成できます。
wavのサンプリングレートやファイル名など、MMVCに適したデータを作成するように作られているので、いくらか手間が省けるかと思います。

### Data作成（なりたい声）
[MMVC](https://github.com/isletennos/MMVC_Trainer)のドキュメントに従って、公式サポートされている声データ(.zip形式)を取得してください。データはzip形式のまま、datasetフォルダにおいてください。


### Datasetの中身
上記Data作成を実施すると、次のようなフォルダ構成になると思います。ご確認ください。
```
$ ls dataset -l
合計 1656692
drwxr-xr-x 4 wataru wataru      4096  8月 22 14:31 00_myvoice
-rwx------ 1 wataru wataru  57620200  8月 22 14:18 1225_zundamon.zip
-rwx------ 1 wataru wataru  72992810  8月 22 14:18 344_tsumugi.zip
-rwx------ 1 wataru wataru  55275760  8月 22 14:18 459_methane.zip
-rwx------ 1 wataru wataru  72295236  8月 22 14:18 912_sora.zip

$ ls dataset/00_myvoice/ -l
合計 40
drwxr-xr-x 2 wataru wataru 20480  8月 22 14:32 text
drwxr-xr-x 2 wataru wataru 20480  8月 22 14:31 wav
```


## 起動と実行

```
# 変数設定
$ EXP_NAME=001_exp

# テスト用フォルダ作成
$ sh template.sh $EXP_NAME


docker run -v .:/go/src/app:ro [container id]
USER_ID=$(id -u)
GROUP_ID=$(id -g)
$ USER_ID=$(id -u) docker run -it --gpus all --shm-size=2g \
  -v `pwd`/exp/${EXP_NAME}/dataset:/MMVC_Trainer/dataset \
  -v `pwd`/exp/${EXP_NAME}/logs:/MMVC_Trainer/logs \
  -v `pwd`/exp/${EXP_NAME}/filelists:/MMVC_Trainer/filelists \
  -e LOCAL_UID=$(id -u $USER) \
  -e LOCAL_GID=$(id -g $USER) \
  -p 6008:6006  mmvc_trainer_docker
```

# ビルド
## 前提
このリポジトリではnodeを使っていませんが、ビルドスクリプト呼び出しにnpmを使用しています。
npmをインストールしておいてください。

https://nodejs.org/ja/download/

## Docker
Dockerを使えるようにしておいてください。
WindowsはWSL2上で使えるようにしておいてください。

## ビルド実行
```
$ npm run build:docker
```

