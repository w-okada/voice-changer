VC Helper
----

# 概要
AIを使ったリアルタイムボイスチェンジャー[MMVC](https://github.com/isletennos/MMVC_Trainer)のヘルパーアプリケーションです。


MMVCで必要となる一連の作業（トレーニング用の音声の録音、トレーニング、ボイスチェンジャ）の中で、音声の録音とボイスチェンジャを各種プラットフォームでお手軽に実行できます。

※ トレーニングについては[公式ノートブック](https://github.com/isletennos/MMVC_Trainer)をご利用ください。

[解説動画](https://youtu.be/MOPqnDPqhAU)

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

# (1) レコーダー（トレーニング用音声録音アプリ）
MMVCトレーニング用の音声を簡単に録音できるアプリです。
Github Pages上で実行できるため、ブラウザのみあれば様々なプラットフォームからご利用可能です。
録音したデータは、ブラウザ上に保存されます。外部に漏れることはありません。

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

・Mac版はダウンロードファイルを解凍したのちに、MMVCServerSIOをダブルクリックしてください。開発元を検証できない旨が示される場合は、再度コントロールキーを押してクリックして実行してください。（詳細下記 *1）

・Windows版は、directML版とGPU版を提供しています。環境に応じたzipファイルをダウンロードしてください。ダンロードしたzipファイルを解凍して、MMVCServerSIO.exeを実行してください。

・NvidiaのGPUをお持ちの方はonnxgpuがファイル名に含まれるファイルをご利用ください。多くの場合はonnxgpu_nocudaの方で動きます。環境によって極まれにgpuが認識されない場合があります。その場合はonnxgpu_cudaの方をご利用ください。（サイズが大きく違います。）

・NvidiaのGPUをお持ちでない方はonnxdirectMLが含まれるファイルをご利用ください。多くの場合は、onnxdirectML_nocudaの方で動きます。環境によって極まれにgpuが認識されない場合があります。その場合はonnxgpu_cudaの方をご利用ください。（サイズが大きく違います。）

・リモートからアクセスできるようにする方法など、より詳しくは[こちら](https://zenn.dev/wok/articles/s01_vc001_top)をご覧ください。


### 最新バージョン
- [MMVCServerSIO_mac_onnxcpu_v.1.3.7.0.zip](https://drive.google.com/file/d/1K_ihZ8hxbQq10qrxM1WUfUaj_vY6zwrW/view?usp=sharing) 154MB
- [MMVCServerSIO_win_onnxdirectML_cuda_v.1.3.7.0.zip](https://drive.google.com/file/d/1IJHazaV60ophM6fbmzugZEjulLpBVJUi/view?usp=sharing) 1962MB
- [MMVCServerSIO_win_onnxdirectML_nocuda_v.1.3.7.0.zip](https://drive.google.com/file/d/1_VzdUpiWb8lbIKNppwsFM5pYCAnixOap/view?usp=sharing) 198MB
- [MMVCServerSIO_win_onnxgpu_cuda_v.1.3.7.0.zip](https://drive.google.com/file/d/1uRZHnDq2nVx4oRlXXiqZeE-ZjJlAFx5C/view?usp=sharing) 2057MB
- [MMVCServerSIO_win_onnxgpu_nocuda_v.1.3.7.0.zip](https://drive.google.com/file/d/1DjSCsc_jKaH-TY6qqFbXz7Ya6tS58odb/view?usp=sharing) 293MB


### 過去バージョン
- [MMVCServerSIO_mac_onnxcpu_v.1.3.6.1.zip](https://drive.google.com/file/d/1WTqPEOkZy19-Z149HZqBI31oi7ffg9Rf/view?usp=sharing) 154MB
- [MMVCServerSIO_win_onnxdirectML_cuda_v.1.3.6.1.exe](https://drive.google.com/file/d/17ZDGa9IUJqm9hePbPczHC6zoqSj_d9CM/view?usp=sharing) 1864MB
- [MMVCServerSIO_win_onnxdirectML_nocuda_v.1.3.6.1.exe](https://drive.google.com/file/d/1V9GF_Nn-FhS5BrwensTuzxkctBR4RUZy/view?usp=sharing) 171MB
- [MMVCServerSIO_win_onnxgpu_cudav.1.3.6.1.exe](https://drive.google.com/file/d/138Jlol8ocy4hM12UNEqlQlZ9XphWOyZp/view?usp=sharing) 1948MB
- [MMVCServerSIO_win_onnxgpu_nocudav.1.3.6.1.exe](https://drive.google.com/file/d/1fnVirudwkoNsY5kabG-295OrqgHzvcHf/view?usp=sharing) 255MB

https://user-images.githubusercontent.com/48346627/212569645-e30b7f4e-079d-4504-8cf8-7816c5f40b00.mp4


詳細は、こちらの[Blog](https://zenn.dev/wok/articles/s01_vc001_top)をご確認ください。


*1 本ソフトウェアは開発元の署名しておりません。下記のように警告が出ますが、コントロールキーを押しながらアイコンをクリックすると実行できるようになります。これはAppleのセキュリティポリシーによるものです。実行は自己責任となります。

![image](https://user-images.githubusercontent.com/48346627/212567711-c4a8d599-e24c-4fa3-8145-a5df7211f023.png)


## (2-3) DockerやAnacondaなど環境構築を行った上での利用
本リポジトリをクローンして利用します。WindowsではWSL2の環境構築が必須になります。また、WSL2上でDockerもしくはAnacondaなどの仮想環境の構築が必要となります。MacではAnacondaなどのPythonの仮想環境の構築が必要となります。事前準備が必要となりますが、多くの環境においてこの方法が一番高速で動きます。**<font color="red"> GPUが無くてもそこそこ新しいCPUであれば十分動く可能性があります </font>（下記のリアルタイム性の節を参照）**。

[WSL2とDockerのインストールの解説動画](https://youtu.be/POo_Cg0eFMU)

[WSL2とAnacondaのインストールの解説動画](https://youtu.be/fba9Zhsukqw)

操作方法は[wiki](https://github.com/w-okada/voice-changer/wiki/040_%E3%83%9C%E3%82%A4%E3%82%B9%E3%83%81%E3%82%A7%E3%83%B3%E3%82%B8%E3%83%A3%E3%83%BC)をご覧ください。

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


