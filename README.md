Voice Changer Trainer and Player
----

# 概要
AIを使ったリアルタイムボイスチェンジャー[MMVC](https://github.com/isletennos/MMVC_Trainer)のヘルパーアプリケーションです。

[解説動画](https://youtu.be/MOPqnDPqhAU)

MMVCで必要となる一連の作業（トレーニング用の音声の録音、トレーニング、ボイスチェンジャ）の中で、音声の録音とボイスチェンジャを各種プラットフォームでお手軽に実行できます。

**※ 公式のv1.3.2.0において、[トレーニング用ノートブック](https://github.com/isletennos/MMVC_Trainer)に大幅なユーザビリティ向上がありました。簡単化を目指していたこちらのトレーニング用アプリの役目は終了したと思われますので開発をストップしています（2023/01/10)。今後は公式のトレーニングの利用を推奨します。**

![image](https://user-images.githubusercontent.com/48346627/201169523-836e0f9e-2aca-4023-887c-52ecc219bcca.png)

このアプリケーションを用いることで、以下のことを簡単に行うことができます。

- MMVCトレーニング用の音声録音 
- MMVCモデルを用いたリアルタイムボイスチェンジャー
  - リアルタイム話者切り替え
  - CPU/GPU切り替え
  - リアルタイム/ニアリアルタイム声質変換


本アプリケーションのリアルタイムボイスチェンジャーは、サーバ・クライアント構成で動きます。MMVCのサーバを別のPC上で動かすことで、ゲーム実況など他の負荷の高い処理への影響を抑えながら動かすことができます。

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

# 使用方法
**v.1.3.x(2023/01/10~)でボイスチェンジャーの大幅な変更を行っています。**

# (1) レコーダー（トレーニング用音声録音アプリ）
MMVCトレーニング用の音声を簡単に録音できるアプリです。
Github Pages上で実行できるため、ブラウザのみあれば様々なプラットフォームからご利用可能です。
録音したデータは、ブラウザ上に保存されます。外部に一切漏れることはありません。

[録音アプリ on Github Pages](https://w-okada.github.io/voice-changer/)

[解説動画](https://youtu.be/s_GirFEGvaA)


詳細については[wiki](https://github.com/w-okada/voice-changer/wiki)をご確認ください。


# (2) プレイヤー（ボイスチェンジャーアプリ）
MMVCでボイチェンを行うためのアプリです。

大きく3つの方法でご利用できます。難易度順に次の通りです。
- Google Colaboratoryでの利用
- 事前ビルド済みのBinaryでの利用
- DockerやAnacondaなど環境構築を行った上での利用

本ソフトウェアやMMVCになじみの薄い方は上から徐々に慣れていくとよいと思います。

## (2-1) Google Colaboratoryでの利用
Googleが提供している機械学習プラットフォームColaboratory上で実行できます。
MMVCのモデルをトレーニングが完了している場合、既にColaboratoryを利用していると思いますので、事前準備は必要ありません。ただし、ネットワーク環境やColaboratoryの状況によってボイスチェンジャのタイムラグが大きくなる傾向があります。

- [超簡単バージョン](https://github.com/w-okada/voice-changer/blob/master/VoiceChangerDemo_Simple.ipynb): 事前設定なしでColabから実行できます。 
- [普通バージョン](https://github.com/w-okada/voice-changer/blob/master/VoiceChangerDemo.ipynb): Google Driveと連携してモデルを読み込むことができます。

[解説動画](https://youtu.be/TogfMzXH1T0)

## (2-2) 事前ビルド済みのBinaryでの利用
実行形式のバイナリをダウンロードして実行することができます。
Windows版とMac版を提供しています。事前準備は必要ありません。

・Mac版はダウンロードした後、実行権限を付与してください。

・Windows版は、directML版とGPU版を提供しています。

・NvidiaのGPUをお持ちの方はonnxgpuが含まれるファイルをご利用ください。多くの場合はonnxgpu_nocudaの方で動きます。環境によってはgpuが認識されない場合が稀にあります。その場合はonnxgpu_cudaの方をご利用ください。（サイズが大きく違います。起動時間も遅くなります）

・NvidiaのGPUをお持ちでない方はonnxdirectMLが含まれるファイルをご利用ください。多くの場合は、onnxdirectML_nocudavの方で動きます。環境によってはgpuが認識されない場合が稀にあります。その場合はonnxgpu_cudaの方をご利用ください。（サイズが大きく違います。起動時間も遅くなります）


- [MMVCServerSIO_mac_onnxcpu_v.1.3.6.0](https://drive.google.com/file/d/1Jfxz4NbjK-jt3yMIdC1Jhec9H47RxK6P/view?usp=sharing) 107MB
- [MMVCServerSIO_win_onnxdirectML_cudav.1.3.6.0.exe](https://drive.google.com/file/d/13ojs8VRconmARDGMoQapCVg3H9AG6PAz/view?usp=sharing) 1864MB
- [MMVCServerSIO_win_onnxdirectML_nocudav.1.3.6.0.exe](https://drive.google.com/file/d/1MHJv2sx_AKxG8YrHvHTeksxGO1zsMTZl/view?usp=sharing) 171MB
- [MMVCServerSIO_win_onnxgpu_cudav.1.3.6.0.exe](https://drive.google.com/file/d/1BWNbIliP0hqB4M3lFpTrFtKHPn6z3KNf/view?usp=sharing) 1948MB
- [MMVCServerSIO_win_onnxgpu_nocudav.1.3.6.0.exe](https://drive.google.com/file/d/1IPaZI53KOhl3eVktP4x0GwyqBngFGViS/view?usp=sharing) 255MB

https://user-images.githubusercontent.com/48346627/212490839-9727daff-8629-4bd7-a885-39b9058a7eba.mp4

詳細は、こちらの[Blog](https://zenn.dev/wok/articles/s01_vc001_top)をご確認ください。

(古いボイスチェンジャについては、引き続き[wiki](https://github.com/w-okada/voice-changer/wiki)をご確認ください。)

## (2-3) DockerやAnacondaなど環境構築を行った上での利用
本リポジトリをクローンして利用します。WindowsではWSL2の環境構築が必須になります。また、WSL2上でDockerもしくはAnacondaなどの仮想環境の構築が必要となります。MacではAnacondaなどのPythonの仮想環境の構築が必要となります。事前準備が必要となりますが、多くの環境においてこの方法が一番高速で動きます。

[WSL2とDockerのインストールの解説動画](https://youtu.be/POo_Cg0eFMU)

[WSL2とAnacondaのインストールの解説動画](https://youtu.be/fba9Zhsukqw)

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


[test](https://drive.google.com/file/d/1NXfaClBvGg2GK9dmhjouqL7JCJm-Ngdd/view?usp=sharing)