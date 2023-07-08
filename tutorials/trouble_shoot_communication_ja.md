## トラブルシュート 通信編

音声が全く変換されない場合や、変換後の音声が変な状態になっている場合、音声変換プロセスの中のどこで問題が起こっているかを切り分ける必要があります。

ここでは、どの部分で問題が起こっているかを大まかに切り分ける方法を説明します。

## VC Client の構成と問題の切り分け

<img src="https://user-images.githubusercontent.com/48346627/235551041-6eed4035-5542-47d1-bbd3-31fa7842011b.png" width="720">

VC Client は、図のように GUI(クライアント)が音声をマイクから拾い、サーバで変換を行う構成となっています。

VC Client は図中の３か所で音声がどのような状態になっているかを確認することができます。
正常な状態の音声が録音されている場合は、そこまでの処理はうまくいっていたということになり、それ以降のところで問題を探せばよいことになります（問題の切り分けといいます）。

## 音声の状態の確認方法

### (1)(2)での音声の状態について確認

<img src="https://github.com/w-okada/voice-changer/assets/48346627/f4845f1d-2e1a-49c1-a226-0e50be807f2d" width="720">

Analyzer の Sampling を start させた状態で音声変換を開始してください。ある程度音声を入力した後に Sampling をストップすると in/out に再生ボタンが表示されます。

- in には前述の図の(1)の音声が録音されています。マイクから入力された音声がそのままサーバで録音されているはずなので、ユーザの音声が録音されていれば OK です。
- out には前述の図の(2)の音声が録音されています。AI による変換後の音声が録音されているはずです。

### (3)での音声の状態について確認

<img src="https://github.com/w-okada/voice-changer/assets/48346627/18ddfc2c-beb2-4e7a-8a06-1e00cc6ddb72" width="720">

AudioOutput の output record を start させた状態で音声変換を開始してください。ある程度音声を入力した後に stop すると.wav ファイルがダウンロードされます。この.wav ファイルはサーバから受信した変換後の音声が録音されているはずです。

## 音声の状態の確認後

前述の図の(1)~(3)のどこまで想定された音声が録音されているかを把握したら、想定された音声が録音された場所以降で問題がないかを検討してください。

### (1)での音声の状態がおかしい場合

#### 音声ファイルでの確認

音声ファイルを入力して変換できるか確認してみてください。

例えばこちらのファイルを使用してみてください。

- [sample_jvs001](https://drive.google.com/file/d/142aj-qFJOhoteWKqgRzvNoq02JbZIsaG/view) from [JVS](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)
- [sample_jvs001](https://drive.google.com/file/d/1iCErRzCt5-6ftALcic9w5zXWrzVXryIA/view) from [JVS-MuSiC](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music)

#### マイク入力の確認

マイク入力自体に問題がある可能性があります。録音ソフトなどを用いてマイク入力を確認してみてください。
また、[こちらの録音サイト](https://w-okada.github.io/voice-changer/)は VCClient の姉妹品であり、ほぼ同等のマイク入力処理が行われているため参考になります。(インストール不要。ブラウザのみで動きます。)
