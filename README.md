
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

ボイスチェンジ時にはGPUは必須ではありません。あればより高速にぼいちぇんできるかもしれません。

## トレーニングデータの準備

### Datasetの中身
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

$ docker run -it --gpus all --shm-size=2g \
  -v `pwd`/exp/${EXP_NAME}/dataset:/MMVC_Trainer/dataset \
  -v `pwd`/exp/${EXP_NAME}/logs:/MMVC_Trainer/logs \
  -v `pwd`/exp/${EXP_NAME}/filelists:/MMVC_Trainer/filelists \
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

