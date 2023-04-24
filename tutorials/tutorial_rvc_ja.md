Realtime Voice Changer Client for RVC チュートリアル(v.1.5.2.4)
================================================================
# はじめに
本アプリケーションは、各種音声変換 AI(VC, Voice Conversion)を用いてリアルタイム音声変換を行うためのクライアントソフトウェアです。本ドキュメントでは[RVC(Retrieval-based-Voice-Conversion)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI)に限定した音声変換のためのチュートリアルを行います。

## 注意事項

- 録音については別途行う必要があります。
  - ブラウザ上で実行可能な録音アプリは[録音アプリ on Github Pages](https://w-okada.github.io/voice-changer/)を参照してください。
  - [解説動画](https://youtu.be/s_GirFEGvaA)
- 学習については別途行う必要があります。
  - 自身で学習を行う場合は[RVC(Retrieval-based-Voice-Conversion)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI)で行ってください。
  - [trainingのTIPS](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/training_tips_ja.md)が公開されているので参照してください。

# 起動まで
## HuBERTのインストール
RVCの実行にはHuBERTが必要です。
[このリポジトリ](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main)から`hubert_base.pt`をダウンロードして、バッチファイルがあるフォルダに格納してください。

## GUIの起動
### Windows 版、
ダウンロードした zip ファイルを解凍して、`start_http.bat`を実行してください。

### Mac 版
ダウンロードファイルを解凍したのちに、`startHttp.command`を実行してください。開発元を検証できない旨が示される場合は、再度コントロールキーを押してクリックして実行してください(or 右クリックから実行してください)。

### リモート接続時の注意
リモートから接続する場合は、`.bat`ファイル(win)、`.command`ファイル(mac)の http が https に置き換わっているものを使用してください。

## クライアントの選択
下記のようなLauncher画面が出れば成功です。この画面からRVCを選択してください。

<img src="/tutorials/images/launcher.png" alt="launcher" width="800" loading="lazy">

## RVC用の画面
下記のような画面が出れば成功です。右上の?ボタンから[マニュアル](https://zenn.dev/wok/books/0004_vc-client-v_1_5_1_x)に移動できます。

<img src="/tutorials/images/RVC_GUI.png" alt="launcher" width="800" loading="lazy">

# クイックスタート
日本語版では[マニュアル](https://zenn.dev/wok/books/0004_vc-client-v_1_5_1_x/viewer/003-1_quick-start)が用意されているのでこちらを参照してください。

## GUIの項目の詳細
## server control
### start
startでサーバーを起動、stopでサーバーを終了します

### monitor
リアルタイム変換の状況を示します。

声を出してから変換までのラグは`buf + res秒`です。

#### vol
音声変換後の音量です。

#### buf
音声を切り取る一回の区間の長さ(ms)です。Input Chunkを短くするとこの数値が減ります。

#### res
Input ChunkとExtra Data Lengthを足したデータを変換にかかる時間です。Input ChunkとExtra Data Lengthのいずれでも短くすると数値が減ります。

### Model Info
サーバが保持している情報を取得します。サーバ・クライアント間で情報同期がうまくいってなさそうなときReloadボタンを押してみてください。

### Switch Model
アップロードしたモデルについて切り替えることができます。

## Model Setting
### Model Uploader
enable PyTorchをオンにするとPyTorchのモデル(拡張子がpth)を選ぶことができます。RVCから変換したモデルを使うときはこちらをオンにすると、PyTorchの項目がでてきます。次のバージョンからSlot毎にPyTorchかONNXのどちらかしか選べなくなります。

#### Model Slot
モデルをどの枠にセットするか選べます。セットしたモデルはServer ControlのSwitch Modelで切り替えられます。

#### Onnx(.onnx)
.onnx形式のモデルをここで指定します。これかPyTorch(.pth)は必須です。

#### PyTorch(.pth)
.pth形式のモデルをここで指定します。これかOnnx(.onnx)は必須です。
RVC-WebUIで学習させた場合、`/logs/weights`に入っています。

#### feature(.npy)
HuBERTで抽出した特徴を訓練データに近づける追加機能です。index(.index)とペアで使用します。
RVC-WebUIで学習させた場合、`/logs/weights/total_fea.npy`という名前で保存されています。

#### index(.index)
HuBERTで抽出した特徴を訓練データに近づける追加機能です。feature(.npy)とペアで使用します。
RVC-WebUIで学習させた場合、`/logs/weights/add_XXX.index`という名前で保存されています。

#### half-precision
精度をfloat32かfloat16のどちらで推論するか選べます。
これを選択すると精度を犠牲に高速化できます。
上手く動かない場合はオフにしてください。

#### Default Tune
声のピッチをどれくらい変換するかデフォルトの値を入れます。推論中に変換もできます。以下は設定の目安です。

- 男声→女声　の変換では+12
- 女声→男声　の変換では-12

#### upload
上記の項目を設定した後、押すとmodelを使用できる状態にします。

#### Framework
アップロードしたモデルファイルのどちらを使うか(PyTorchかONNXか)を選びます。次のバージョンではなくなる予定です。

## Device Setting
### AudioInput
入力端末を選びます

### AudioOutput
出力端末を選びます

#### output record
startをおしてからstopを押すまでの音声が記録されます。
このボタンを押してもリアルタイム変換は始まりません。
リアルタイム変換はServer Controlを押してください

## Quality Control

### Noise Supression
ブラウザ組み込みのノイズ除去機能のOn/Offです。

### Gain Control
- input:モデルへの入力音声の音量を増減します。１がデフォルト
- output:モデルからの出力音声の音量を増減します。１がデフォルト

### F0Detector
ピッチを抽出するためのアルゴリズムを選びます。以下の二種類を選べます。

- 軽量な`pm`
- 高精度な`harvest`

### Analyzer(Experimental)
サーバ側で入力と出力を録音します。
入力はマイクの音声がサーバに送られて、それがそのまま録音されます。マイク⇒サーバの通信路の確認に使えます。
出力はモデルから出力されるデータがサーバ内で録音されます。(入力が正しいことが確認できたうえで)モデルの動作を確認できます。


## Speaker Setting
### Destination Speaker Id
複数話者に対応した時の設定かと思われますが、RVC本家が対応していない(見込みもない)ので現状は使わない項目です。

### Tuning
声のピッチを調整します。以下は設定の目安です。

- 男声→女声　の変換では+12
- 女声→男声　の変換では-12

### index ratio
学習で使用した特徴量に寄せる比率を指定します。Model Settingでfeatureとindexを両方設定した時に有効です。
0でHuBERTの出力をそのまま使う、1で元の特徴量にすべて寄せます。
index ratioが0より大きいと検索に時間がかかる場合があります。

### Silent Threshold
音声変換を行う音量の閾地です。この値より小さいrmsの時は音声変換をせず無音を返します。
（この場合、変換処理がスキップされるので、あまり負荷がかかりません。）

## Converter Setting
### InputChunk Num(128sample / chunk)
一度の変換でどれくらいの長さを切り取って変換するかを決めます。これが大きいほど効率的に変換できますが、bufの値が大きくなり変換が開始されるまでの最大の時間が伸びます。 buff: におよその時間が表示されます。

### Extra Data Length
音声を変換する際、入力にどれくらいの長さの過去の音声を入れるかを決めます。過去の音声が長く入っているほど変換の精度はよくなりますが、その分計算に時間がかかるためresが長くなります。
(おそらくTransformerがネックなので、これの長さの2乗で計算時間は増えます)

### GPU
GPUを2枚以上持っている場合、ここでGPUを選べます。
