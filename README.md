Voice Changer Trainer and Player
----
# News
2022/12/09: Anacondaに対応しました(β版)。以下の構成に対応します。

| #   | os            | middle   |
| --- | ------------- | -------- |
| 1   | Windows       | Anaconda |
| 2   | Windows(WSL2) | Docker   |
| 3   | Windows(WSL2) | Anaconda |
| 4   | Mac(Intel)    | Anaconda |
| 5   | Mac(M1)       | Anaconda |
| 6   | Linux         | Docker   |
| 7   | Linux         | Anaconda |
| 8   | Colab         | Notebook |

# 概要
AIを使ったリアルタイムボイスチェンジャー[MMVC](https://github.com/isletennos/MMVC_Trainer)のヘルパーアプリケーションです。

MMVCで必要となる一連の作業（トレーニング用の音声の録音、トレーニング、ボイスチェンジャ）をお手軽に実行できます。
![image](https://user-images.githubusercontent.com/48346627/201169523-836e0f9e-2aca-4023-887c-52ecc219bcca.png)

このアプリケーションを用いることで、以下のことを簡単に行うことができます。

- MMVCトレーニング用の音声録音 (GithubPages (ローカル環境構築不要))
- MMVCのモデルのトレーニング (Dockerを強く推奨、その他構成Colabでも可)
- MMVCモデルを用いたリアルタイムボイスチェンジャー
  - リアルタイム話者切り替え
  - CPU/GPU切り替え
  - リアルタイム/ニアリアルタイム声質変換


本アプリケーションのリアルタイムボイスチェンジャーは、サーバ・クライアント構成で動きます。MMVCのサーバを別のPC上で動かすことで、ゲーム実況など他の負荷の高い処理への影響を抑えながら動かすことができます。

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

# 使用方法

以下では、使用方法のダイジェストのみ記載します。詳細な使用方法等は[wiki](https://github.com/w-okada/voice-changer/wiki)をご参照ください。


## レコーダー（トレーニング用音声録音アプリ）

MMVCトレーニング用の音声を簡単に録音できるアプリです。
Github Pages上で実行できるため、Dockerの準備は不要です。

[録音アプリ on Github Pages](https://w-okada.github.io/voice-changer/)

録音したデータは、ブラウザ上に保存されます。外部に一切漏れることはありません。

## トレーナー（トレーニング用アプリ）
MMVCのトレーニングを行うアプリです。
Dockerでの実行を強く推奨します。

Colaboratoryでの実行を行う場合は、次のノートから実行できます。
- [Colab ノート](https://github.com/w-okada/voice-changer/blob/master/MMVCTrainerFrontendDemo.ipynb)

Colaboratoryで実行する場合は、Colabの制約により途中切断が発生する可能性があります。Web上では各種対策が示されていますので、必要に応じて各自でご対応をお願いします。また、途中切断に備え、こまめなモデルのダウンロードをお勧めします。

## プレイヤー（ボイスチェンジャーアプリ）
MMVCでボイチェンを行うためのアプリです。
お手元のPCでの使用を推奨します。一部ユーザ体験が劣化しますが、次のノートでColaboratoryでの実行も可能です。
- [超簡単バージョン](https://github.com/w-okada/voice-changer/blob/master/VoiceChangerDemo_Simple.ipynb): 事前設定なしでColabから実行できます。 
- [普通バージョン](https://github.com/w-okada/voice-changer/blob/master/VoiceChangerDemo.ipynb): Google Driveと連携してモデルを読み込むことができます。

# 説明動画
| No  | タイトル                             | リンク                                  |
| --- | ------------------------------------ | --------------------------------------- |
| 01  | ざっくり説明編                       | [youtube](https://youtu.be/MOPqnDPqhAU) |
| 02  | ユーザー音声の録音編                 | [youtube](https://youtu.be/s_GirFEGvaA) |
| 03  | トレーニング編                       | 作成中                                  |
| 04a | Colabでボイチェン編                  | [youtube](https://youtu.be/TogfMzXH1T0) |
| 04b | PCでボイチェン編                     | 欠番(ex1, ex2, 04aの内容と被るため)     |
| ex1 | 番外編：WSL2とDockerのインストール   | [youtube](https://youtu.be/POo_Cg0eFMU) |
| ex2 | 番外編：WSL2とAnacondaのインストール | [youtube](https://youtu.be/fba9Zhsukqw) |


## リアルタイム性

GPUを使用すると100msec強の遅延で変換することが可能です。

https://user-images.githubusercontent.com/48346627/199807082-9d2ca75b-3a05-463d-b32e-14e663603626.mp4


CPUでも最近のであればそれなりの速度で変換可能。

https://user-images.githubusercontent.com/48346627/199855821-596f0874-8f69-4354-b226-4a755e9763bc.mp4


古いCPU( i7-4770)だと、1000msecくらいかかってしまう。

# Misc
## 簡易デモ(soft-vc)
soft-vcを用いたボイスチェンジャです。

[コラボのノート](https://github.com/w-okada/voice-changer/blob/master/SoftVcDemo.ipynb)

[説明動画](https://user-images.githubusercontent.com/48346627/191019809-e7ae7c86-4b44-45f3-9dc3-3dc668992db4.mp4
)



# Acknowledgments
- 立ちずんだもん素材：https://seiga.nicovideo.jp/seiga/im10792934
- いらすとや：https://www.irasutoya.com/
