Voice Changer Trainer and Player Container
----
# 概要
AIを使ったリアルタイムボイスチェンジャー[MMVC](https://github.com/isletennos/MMVC_Trainer)のトレーニングと実行を簡単にするためのヘルパーDockerコンテナです。
このコンテナを用いることで、以下のことを簡単に行うことができます。

- MMVCトレーニング用の音声録音
- MMVCのモデルのトレーニング
- MMVCモデルを用いたリアルタイム声質変換（ボイスチェンジャ）
  - リアルタイム話者切り替え
  - CPU/GPU切り替え
  - リアルタイム/ニアリアルタイム声質変換

このコンテナのボイスチェンジャは、サーバ・クライアント構成で動きます。MMVCのサーバを別のPC上で動かすことで、ゲーム実況など他の不可の高い処理への影響を抑えながら動かすことができます。（MacのChromeからも利用できます！！）
![image](https://user-images.githubusercontent.com/48346627/193464403-ca981f72-6186-4eda-b715-55abdf236b17.png)


## 使用方法

使用方法等は[wiki](https://github.com/w-okada/voice-changer/wiki)をご参照ください。

## MMVCのトレーニング用音声データと、実行時の音声入力の考え方
なお、「MMVCトレーニング用の音声録音」と「MMVCモデルを用いたリアルタイム声質変換」では同一のノイズキャンセル技術が使用されています。
一般にMMVCなどのAI/機械学習を用いたアプリケーションでは、学習データと似たデータをボイスチェンジャに入力することが望ましいとされます。「MMVCトレーニング用の音声録音」と「MMVCモデルを用いたリアルタイム声質変換」を本コンテナに統一することで、実行時の精度が向上する可能性があります。

![image](https://user-images.githubusercontent.com/48346627/191024059-9c90dfbc-8098-4a81-a905-2a8aa51662ba.png)


## 関連技術
本レポジトリでは、関連するボイスチェンジャとして[soft-vc](https://github.com/bshall/soft-vc)に注目しています。soft-vcにもリアルタイムでボイスチェンジできるようにしています。下記の「Docker不要な実行方法」のセクションからご利用ください。


# Docker不要な実行方法

本コンテナの一部の機能は、Google ColabやGithub Pagesを利用することで、ローカルPCにDockerの環境を用意することなく利用可能になっています。

お気軽に利用することが可能ですので、試しに使用してみてください。

## トレーニング用音声録音アプリ

MMVCトレーニング用の音声を簡単に録音できるアプリです。
Github Pagesにおいてあります。下記リンクにアクセスすることで利用可能です。

[録音アプリ on Github Pages](https://w-okada.github.io/voice-changer/)

録音したデータは、ブラウザ上に保存されます。外部に一切漏れることはありません。


また、Google Colab上にサーバを立てて利用することもできます。
こちらを使うと、録音データをGoogle Drive上に保存することができるようになります。MMVCのトレーニングをColab上で行う場合は、こちらを使用すると多少の手間が省けます。
[コラボのノート](https://github.com/w-okada/voice-changer/blob/master/VoiceRecorder.ipynb)


使用方法は[wiki](https://github.com/w-okada/voice-changer/wiki/500_%E3%83%AC%E3%82%B3%E3%83%BC%E3%83%80%E3%83%BC)をご覧ください。


## 簡易デモ(MMVC)
MMVCを用いたボイスチェンジャです。

[コラボのノート](https://github.com/w-okada/voice-changer/blob/dev/VoiceChangerDemo.ipynb)

[説明動画](https://twitter.com/DannadoriYellow/status/1564897136999022592)

動画との差分

- サーバの起動完了のメッセージは、「Debuggerほにゃらら」ではなく「Application startup complete.」です。
- プロキシにアクセスする際に、index.htmlを追加する必要はありません。

## 簡易デモ(soft-vc)
soft-vcを用いたボイスチェンジャです。

[コラボのノート](https://github.com/w-okada/voice-changer/blob/master/SoftVcDemo.ipynb)

[説明動画](https://user-images.githubusercontent.com/48346627/191019809-e7ae7c86-4b44-45f3-9dc3-3dc668992db4.mp4
)


