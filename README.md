Voice Changer Trainer and Player
----

# 概要
AIを使ったリアルタイムボイスチェンジャー[MMVC](https://github.com/isletennos/MMVC_Trainer)のヘルパーアプリケーションです。


MMVCで必要となる一連の作業（トレーニング用の音声の録音、トレーニング、ボイスチェンジャ）の中で、音声の録音とボイスチェンジャを各種プラットフォームでお手軽に実行できます。

**※ [公式のトレーニング用ノートブック](https://github.com/isletennos/MMVC_Trainer)に大幅なユーザビリティ向上がありました。簡単化を目指していたこちらのトレーニング用アプリの役目は終了したと思われますので開発をストップしています（2023/01/10)。公式のトレーニングの利用を推奨します。**

![image](https://user-images.githubusercontent.com/48346627/201169523-836e0f9e-2aca-4023-887c-52ecc219bcca.png)

このアプリケーションを用いることで、以下のことを簡単に行うことができます。

- MMVCトレーニング用の音声録音 (GithubPages (ローカル環境構築不要))
- MMVCモデルを用いたリアルタイムボイスチェンジャー
  - リアルタイム話者切り替え
  - CPU/GPU切り替え
  - リアルタイム/ニアリアルタイム声質変換


本アプリケーションのリアルタイムボイスチェンジャーは、サーバ・クライアント構成で動きます。MMVCのサーバを別のPC上で動かすことで、ゲーム実況など他の負荷の高い処理への影響を抑えながら動かすことができます。

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

# 使用方法
**v.1.3.x(2023/01/10~)でボイスチェンジャーの大幅な変更を行っています。**

## レコーダー（トレーニング用音声録音アプリ）

MMVCトレーニング用の音声を簡単に録音できるアプリです。
Github Pages上で実行できるため、ブラウザのみあれば様々なプラットフォームからご利用可能です。

[録音アプリ on Github Pages](https://w-okada.github.io/voice-changer/)

録音したデータは、ブラウザ上に保存されます。外部に一切漏れることはありません。

詳細については引き続き[wiki](https://github.com/w-okada/voice-changer/wiki)をご確認ください。


## プレイヤー（ボイスチェンジャーアプリ）
MMVCでボイチェンを行うためのアプリです。
お手元のPCでの使用を推奨します。一部ユーザ体験が劣化しますが、次のノートでColaboratoryでの実行も可能です。
- [超簡単バージョン](https://github.com/w-okada/voice-changer/blob/master/VoiceChangerDemo_Simple.ipynb): 事前設定なしでColabから実行できます。 
- [普通バージョン](https://github.com/w-okada/voice-changer/blob/master/VoiceChangerDemo.ipynb): Google Driveと連携してモデルを読み込むことができます。

また、バイナリ（α版）の配布も行っています。M1 Mac版は、ダウンロード後、ターミナルで実行権限を付与して実行してください。

- [M1 Mac v.1.3.5α](https://drive.google.com/file/d/1UJhitp0uZAhcQmcdve-iirDws4iMfD74/view?usp=sharing)


- [Win  v.1.3.5α](https://drive.google.com/file/d/1UOEt3l4oxCsePOILChNG3B-yhHciHkQY/view?usp=sharing)

- [Wind v.1.3.5DMLα](https://drive.google.com/file/d/1lJex64Y6RwkTTs72xB-pg9dQLjSuWi83/view?usp=sharing)

https://user-images.githubusercontent.com/48346627/212490839-9727daff-8629-4bd7-a885-39b9058a7eba.mp4


詳細は、こちらの[Blog](https://zenn.dev/wok/articles/s01_vc001_top)をご確認ください。

(古いボイスチェンジャについては、引き続き[wiki](https://github.com/w-okada/voice-changer/wiki)をご確認ください。)



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

GPUを使用するとほとんどタイムラグなく変換可能です。

https://twitter.com/DannadoriYellow/status/1613483372579545088?s=20&t=7CLD79h1F3dfKiTb7M8RUQ

CPUでも最近のであればそれなりの速度で変換可能。

https://twitter.com/DannadoriYellow/status/1613553862773997569?s=20&t=7CLD79h1F3dfKiTb7M8RUQ

古いCPU( i7-4770)だと、1000msecくらいかかってしまう。

# Acknowledgments
- 立ちずんだもん素材：https://seiga.nicovideo.jp/seiga/im10792934
- いらすとや：https://www.irasutoya.com/

# 免責事項
本ソフトウェアの使用または使用不能により生じたいかなる直接損害・間接損害・波及的損害・結果的損害 または特別損害についても、一切責任を負いません。