# Realtime Voice Changer Client for RVC チュートリアル(v.1.5.3.1)

# はじめに

本アプリケーションは、各種音声変換 AI(VC, Voice Conversion)を用いてリアルタイム音声変換を行うためのクライアントソフトウェアです。本ドキュメントでは[RVC(Retrieval-based-Voice-Conversion)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI)に限定した音声変換のためのチュートリアルを行います。

以下、本家の[Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)を本家 RVC と表記し、ddPn08 氏の作成した[RVC-WebUI](https://github.com/ddPn08/rvc-webui)を ddPn08RVC と記載します。

## 注意事項

- 学習については別途行う必要があります。
  - 自身で学習を行う場合は[本家 RVC](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI)または[ddPn08RVC](https://github.com/ddPn08/rvc-webui)で行ってください。
  - ブラウザ上で学習用の音声を用意するには[録音アプリ on Github Pages](https://w-okada.github.io/voice-changer/)が便利です。
    - [解説動画](https://youtu.be/s_GirFEGvaA)
  - [training の TIPS](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/training_tips_ja.md)が公開されているので参照してください。

# 起動まで

## GUI の起動

### Windows 版、

ダウンロードした zip ファイルを解凍して、`start_http.bat`を実行してください。

### Mac 版

ダウンロードファイルを解凍したのちに、`startHttp.command`を実行してください。開発元を検証できない旨が示される場合は、再度コントロールキーを押してクリックして実行してください(or 右クリックから実行してください)。

### リモート接続時の注意

リモートから接続する場合は、`.bat`ファイル(win)、`.command`ファイル(mac)の http が https に置き換わっているものを使用してください。

### コンソール表示

`.bat`ファイル(win)や`.command`ファイル(mac)を実行すると、次のような画面が表示され、初回起動時には各種データをインターネットからダウンロードします。
お使いの環境によりますが、多くの場合１～２分かかります。
![image](https://github.com/w-okada/voice-changer/assets/48346627/88a30097-2fb3-4c50-8bf1-19c41f27c481)

### GUI 表示

起動に必要なデータのダウンロードが完了すると下記のような Launcher 画面が出ます。この画面から RVC を選択してください。

![クライアントの選択画面](https://user-images.githubusercontent.com/23290400/235131650-9eeee978-96fa-478a-b728-3581ae0b8b67.png)

## RVC 用の画面

下記のような画面が出れば成功です。右上の?ボタンから[マニュアル](https://zenn.dev/wok/books/0004_vc-client-v_1_5_1_x)に移動できます。

![v1.5.3.1 RVC初期画面](https://github.com/w-okada/voice-changer/assets/48346627/0f407779-7798-49f9-a542-663d80807cdb)

# クイックスタート

起動時にダウンロードしたデータを用いて、すぐに音声変換を行うことができます。

下図の(1)で使用するマイクとスピーカーを選択して、(2)のスタートボタンを押してみてください。
数秒のデータロードの後に音声変換が開始されます。
なお、慣れていない方は、(1)では client device を選択してかマイクとスピーカーを選択することを推奨します。（server device との違いは後述します。）

![image](https://github.com/w-okada/voice-changer/assets/48346627/ce2f8be7-852e-4b78-adce-1df8cad9fbab)

## GUI の項目の詳細

GUI で設定できる項目は下図のようなセクションに分かれています。それぞれのセクションはタイトルをクリックすることで開閉できます。

![image](https://github.com/w-okada/voice-changer/assets/48346627/a5eab90c-c0af-42cd-abfb-e897d333d1ff)

## server control

### start

start でサーバーを起動、stop でサーバーを停止します

### monitor

リアルタイム変換の状況を示します。

声を出してから変換までのラグは`buf + res秒`です。調整の際は buf の時間が res よりも長くなるように調整してください。

なお、デバイスを server device モードで使用している場合はこの表示は行われません。コンソール側に表示されます。

#### vol

音声変換後の音量です。

#### buf

音声を切り取る一回の区間の長さ(ms)です。Input Chunk を短くするとこの数値が減ります。

#### res

Input Chunk と Extra Data Length を足したデータを変換にかかる時間です。Input Chunk と Extra Data Length のいずれでも短くすると数値が減ります。

### Switch Model

アップロードしたモデルについて切り替えることができます。
モデルについては名前の下に[]で情報が示されます

1. f0(=pitch)を考慮するモデルか

- f0: 考慮する
- nof0: 考慮しない

2. モデルの学習に用いられたサンプリングレート
3. モデルが用いる特徴量のチャンネル数(大きいほど情報を持っていて重い)
4. 学習に用いられたクライアント

- org: [本家 RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)で学習したモデルです。
- webui:[ddPn08RVC](https://github.com/ddPn08/rvc-webui)で学習したモデルです。

### Operation

モデル、サーバに対する処理を実行するボタンが配置されています。

#### export onnx

ONNX モデルを出力します。PyTorch のモデルを ONNX モデルに変換すると、推論が高速化される場合があります。

#### download

モデルをダウンロードします。主にモデルマージした結果を取得するために使います。

## Model Setting

#### Model Slot

モデルをどの枠にセットするか選べます。セットしたモデルは Server Control の Switch Model で切り替えられます。

モデルをセットする際に、ファイルから読み込むか、インターネットからダウンロードするかを選択できます。この選択結果に応じて設定できる項目が変化します。

- file: ローカルファイルを選択してモデルを読み込みます。
- from net: インターネットからモデルをダウンロードします。

#### Model(.onnx or .pth)

ファイルから読み込む設定にした場合に表示されます。

学習済みモデルをここで指定します。必須項目です。
ONNX 形式(.onnx)か PyTorch 形式(.pth)のいずれかを選択可能です。

- [orginal-RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)で学習させた場合、`/logs/weights`に入っています。
- [ddPn08RVC](https://github.com/ddPn08/rvc-webui)で学習させた場合、`/models/checkpoints`に入っています。

#### feature(.npy)

ファイルから読み込む設定にした場合に表示されます。

HuBERT で抽出した特徴を訓練データに近づける追加機能です。index(.index)とペアで使用します。

- [orginal-RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)で学習させた場合、`/logs/実験名/total_fea.npy`という名前で保存されています。(2023/04/26 に total_fea.npy を省略するアップデートが入ったので今後不要になる可能性があります)
- [ddPn08RVC](https://github.com/ddPn08/rvc-webui)で学習させた場合、`/models/checkpoints/モデル名_index/モデル名.0.big.npy`という名前で保存されています。

#### index(.index)

ファイルから読み込む設定にした場合に表示されます。

HuBERT で抽出した特徴を訓練データに近づける追加機能です。feature(.npy)とペアで使用します。

- [orginal-RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)で学習させた場合、`/logs/実験名/add_XXX.index`という名前で保存されています。
- [ddPn08RVC](https://github.com/ddPn08/rvc-webui)で学習させた場合、`/models/checkpoints/モデル名_index/モデル名.0.index`という名前で保存されています。

#### Select Model

インターネットからダウロードする設定にした場合に表示されます。

ダウンロードするモデルを選択します。利用規約へのリンクが表示されるので、ご使用の際にはご確認ください。

#### Default Tune

声のピッチをどれくらい変換するかデフォルトの値を入れます。推論中に変換もできます。以下は設定の目安です。

- 男声 → 女声　の変換では+12
- 女声 → 男声　の変換では-12

#### upload

ファイルから読み込む設定にした場合に表示されます。

上記の項目を設定した後、押すと model を使用できる状態にします。

#### select

インターネットからダウロードする設定にした場合に表示されます。

上記の項目を設定した後、押すと model を使用できる状態にします。

## Speaker Setting

### Tuning

声のピッチを調整します。以下は設定の目安です。

- 男声 → 女声　の変換では+12
- 女声 → 男声　の変換では-12

### index ratio

学習で使用した特徴量に寄せる比率を指定します。Model Setting で feature と index を両方設定した時に有効です。
0 で HuBERT の出力をそのまま使う、1 で元の特徴量にすべて寄せます。
index ratio が 0 より大きいと検索に時間がかかる場合があります。

### Silent Threshold

音声変換を行う音量の閾地です。この値より小さい rms の時は音声変換をせず無音を返します。
（この場合、変換処理がスキップされるので、あまり負荷がかかりません。）

## Converter Setting

### InputChunk Num(128sample / chunk)

一度の変換でどれくらいの長さを切り取って変換するかを決めます。これが大きいほど効率的に変換できますが、buf の値が大きくなり変換が開始されるまでの最大の時間が伸びます。 buff: におよその時間が表示されます。

### Extra Data Length

音声を変換する際、入力にどれくらいの長さの過去の音声を入れるかを決めます。過去の音声が長く入っているほど変換の精度はよくなりますが、その分計算に時間がかかるため res が長くなります。
(おそらく Transformer がネックなので、これの長さの 2 乗で計算時間は増えます)

詳細は[こちらの資料](https://github.com/w-okada/voice-changer/issues/154#issuecomment-1502534841)をご覧ください。

### GPU

GPU を 2 枚以上持っている場合、ここで GPU を選べます。

## Device Setting

### Device Mode

client device mode と server device mode のどちらを使用するか選択します。音声変換が停止している時のみ変更できます。

それぞれのモードの詳細は[こちら](./tutorial_device_mode.md)をご覧ください。

### AudioInput

入力端末を選びます

### AudioOutput

出力端末を選びます

### output record

client device mode の時のみ表示されます。

start をおしてから stop を押すまでの音声が記録されます。
このボタンを押してもリアルタイム変換は始まりません。
リアルタイム変換は Server Control を押してください

## Lab

モデルマージを行うことができます。

各マージ元モデルの成分量を設定します。成分量の比率に従って新しいモデルを生成します。

## Quality Control

### Noise Supression

ブラウザ組み込みのノイズ除去機能の On/Off です。

### Gain Control

- input:モデルへの入力音声の音量を増減します。１がデフォルト
- output:モデルからの出力音声の音量を増減します。１がデフォルト

### F0Detector

ピッチを抽出するためのアルゴリズムを選びます。以下の二種類を選べます。

- 軽量な`pm`
- 高精度な`harvest`

### Analyzer(Experimental)

サーバ側で入力と出力を録音します。
入力はマイクの音声がサーバに送られて、それがそのまま録音されます。マイク ⇒ サーバの通信路の確認に使えます。
出力はモデルから出力されるデータがサーバ内で録音されます。(入力が正しいことが確認できたうえで)モデルの動作を確認できます。
