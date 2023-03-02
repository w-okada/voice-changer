VC Helper
----
# VC Helperとは
[VC Helper](https://github.com/w-okada/voice-changer)はAIを使ったリアルタイムボイスチェンジャー[MMVC](https://github.com/isletennos/MMVC_Trainer)のヘルパーアプリケーションです。MMVCで必要となるトレーニング用の音声の録音とボイスチェンジャを各種プラットフォームでお手軽に実行できます。
[解説動画](https://www.nicovideo.jp/watch/sm41507891)

※ トレーニングについては[公式ノートブック](https://github.com/isletennos/MMVC_Trainer)をご利用ください。

![image](https://user-images.githubusercontent.com/48346627/201169523-836e0f9e-2aca-4023-887c-52ecc219bcca.png)


# 特徴
1. 複数のプラットフォームで動作
Windows, Mac(M1等Apple silicon), Linux, Google Colaboratoryでの動作をサポートしています。

2. 音声録音用アプリのインストールが不要
音声録音をGithub Pagesにホストしてあるアプリケーション上で実行可能です。全てブラウザ上で動くため、特別なアプリケーションのインストールは不要です。また、完全にブラウザアプリケーションとして動くのでデータがサーバに送信されることもありません。

3. ボイチェンを別のPC上で実行して負荷を分散
本アプリケーションのリアルタイムボイスチェンジャーは、サーバ・クライアント構成で動きます。MMVCのサーバを別のPC上で動かすことで、ゲーム実況など他の負荷の高い処理への影響を抑えながら動かすことができます。

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

# 使用方法
詳細は[こちら(v.1.3.x)](https://zenn.dev/wok/books/0002_vc-helper-v_1_3)に纏まっています。
(v1.5.xは[こちら](https://zenn.dev/wok/books/0003_vc-helper-v_1_5))

# (1) レコーダー（トレーニング用音声録音アプリ）
MMVCトレーニング用の音声を簡単に録音できるアプリです。
Github Pages上で実行できるため、ブラウザのみあれば様々なプラットフォームからご利用可能です。
録音したデータは、ブラウザ上に保存されます。外部に漏れることはありません。

[録音アプリ on Github Pages](https://w-okada.github.io/voice-changer/)

[解説動画](https://youtu.be/s_GirFEGvaA)


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


・Mac版はダウンロードファイルを解凍したのちに、MMVCServerSIOをダブルクリックしてください。開発元を検証できない旨が示される場合は、再度コントロールキーを押してクリックして実行してください(or 右クリックから実行してください)。（詳細下記 *1）

・Windows版は、directML版とGPU版を提供しています。環境に応じたzipファイルをダウンロードしてください。ダウンロードファイルしたzipファイルを解凍して、MMVCServerSIO.exeを実行してください。

・NvidiaのGPUをお持ちの方はonnxgpuがファイル名に含まれるファイルをご利用ください。多くの場合はonnxgpu_nocudaの方で動きます。環境によって極まれにgpuが認識されない場合があります。その場合はonnxgpu_cudaの方をご利用ください。（サイズが大きく違います。）

・NvidiaのGPUをお持ちでない方はonnxdirectMLが含まれるファイルをご利用ください。多くの場合は、onnxdirectML_nocudaの方で動きます。環境によって極まれにgpuが認識されない場合があります。その場合はonnxgpu_cudaの方をご利用ください。（サイズが大きく違います。）

・リモートからアクセスできるようにする方法など、より詳しくは[こちら](https://zenn.dev/wok/books/0002_vc-helper-v_1_3)をご覧ください。

### アルファ版(for v.1.5.x)
- [MMVCServerSIO_mac_onnxcpu_v.1.5.1.4a.zip](https://drive.google.com/file/d/1urqcB_S4lqbrxL4osKIlQ6MLhsh__W7t/view?usp=sharing) 510MB
- [MMVCServerSIO_win_onnxdirectML_cuda_v.1.5.1.4a.zip](https://drive.google.com/file/d/1KSmmu5A29f3wXc_ZreycuuCyLltgNg5h/view?usp=sharing) 2.45GB
- [MMVCServerSIO_win_onnxdirectML_nocuda_v.1.5.1.4a.zip](https://drive.google.com/file/d/1B_dPZMIf39Of7olTVzR0h6fNP5u0lx8P/view?usp=sharing) 430MB
- [MMVCServerSIO_win_onnxgpu_cuda_v.1.5.1.4a.zip](https://drive.google.com/file/d/1sUa42la2vjTkIMcLKRSBf8icvfB6fqM0/view?usp=sharing) 2.55GB
- [MMVCServerSIO_win_onnxgpu_nocuda_v.1.5.1.4a.zip](https://drive.google.com/file/d/1cQVnwenJD0vXzyThZ3iiMBUVLRXoDPBJ/view?usp=sharing) 541MB

### 最新バージョン(for v.1.3.x)
- [MMVCServerSIO_mac_onnxcpu_v.1.3.9.4.zip](https://drive.google.com/file/d/1dliqQE7Kn5vhycrDUZQ6pgwLfP_znAyp/view?usp=sharing) 510MB
- [MMVCServerSIO_win_onnxdirectML_cuda_v.1.3.9.4.zip](https://drive.google.com/file/d/1vfZc52f0BVD8nGjsuaAhl0jb4djHBBYe/view?usp=sharing) 2.45GB
- [MMVCServerSIO_win_onnxdirectML_nocuda_v.1.3.9.4.zip](https://drive.google.com/file/d/14DaEPJnio-Ne50e2t1wFRgwQYnWwkh6n/view?usp=sharing) 430MB
- [MMVCServerSIO_win_onnxgpu_cuda_v.1.3.9.4.zip](https://drive.google.com/file/d/1wBgjNHf0Kz3BPp-73KCbCEnPTW4FnuEE/view?usp=sharing) 2.55GB
- [MMVCServerSIO_win_onnxgpu_nocuda_v.1.3.9.4.zip](https://drive.google.com/file/d/16R44mbi4AlkynVzhROrmA6u9MRSegpwd/view?usp=sharing) 541MB


### 過去バージョン
ページの一番下にまとめてあります。

https://user-images.githubusercontent.com/48346627/212569645-e30b7f4e-079d-4504-8cf8-7816c5f40b00.mp4


*1 本ソフトウェアは開発元の署名しておりません。下記のように警告が出ますが、コントロールキーを押しながらアイコンをクリックすると実行できるようになります。これはAppleのセキュリティポリシーによるものです。実行は自己責任となります。

![image](https://user-images.githubusercontent.com/48346627/212567711-c4a8d599-e24c-4fa3-8145-a5df7211f023.png)


## (2-3) DockerやAnacondaなど環境構築を行った上での利用
本リポジトリをクローンして利用します。WindowsではWSL2の環境構築が必須になります。また、WSL2上でDockerもしくはAnacondaなどの仮想環境の構築が必要となります。MacではAnacondaなどのPythonの仮想環境の構築が必要となります。事前準備が必要となりますが、多くの環境においてこの方法が一番高速で動きます。**<font color="red"> GPUが無くてもそこそこ新しいCPUであれば十分動く可能性があります </font>（下記のリアルタイム性の節を参照）**。

[WSL2とDockerのインストールの解説動画](https://youtu.be/POo_Cg0eFMU)

[WSL2とAnacondaのインストールの解説動画](https://youtu.be/fba9Zhsukqw)

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


### 過去バージョン(for v.1.5.x)
- [MMVCServerSIO_mac_onnxcpu_v.1.5.0.8a.zip](https://drive.google.com/file/d/1HhgrPMQwgjVgJJngsyZ4JJiieQQAI-zC/view?usp=sharing) 509MB
- [MMVCServerSIO_win_onnxgpu_cuda_v.1.5.0.8a.zip](https://drive.google.com/file/d/182q30PeI7ULgdtn-wg5VEGb0mUfHsCi5/view?usp=sharing)2.55GB
- [MMVCServerSIO_mac_onnxcpu_v.1.5.0.6a.zip](https://drive.google.com/file/d/1x2NOPqe9dOOjLtzsElgNs60vUWzdPd5f/view?usp=sharing)
- [MMVCServerSIO_win_onnxgpu_cuda_v.1.5.0.6a.zip](https://drive.google.com/file/d/1K9Q5QPzTZJRHsY1KXNc8JeY2K6jFF0hs/view?usp=sharing)
- [MMVCServerSIO_win_cuda_v.1.5.0.5a.zip](https://drive.google.com/file/d/1FdvkRfSevcPrig2A_BBZ70AamDVHXbpa/view?usp=sharing) 2.5GB
- [MMVCServerSIO_mac_cpu_v.1.5.0.5a.zip](https://drive.google.com/file/d/18DY4aXhLqaCcPIfGZqPFCsIt30jEZw24/view?usp=sharing) 509MB
- [MMVCServerSIO_win_cuda_v.1.5.0.4a.zip](https://drive.google.com/file/d/1FIOds2TuNjdw2XNNWNdN1OSwCK9svS6b/view?usp=sharing) 2.5GB
- [MMVCServerSIO_mac_cpu_v.1.5.0.4a.zip](https://drive.google.com/file/d/10JHbED9Vj5Y-zJIBtag_XXMHE9-MhAyF/view?usp=sharing) 509MB
- [MMVCServerSIO_win_cuda_v.1.5.0.3a.zip](https://drive.google.com/file/d/1WxUyZ69Hbuu6p6TBvW0DX0Qt2a-2NQlw/view?usp=sharing) 2.5GB
- [MMVCServerSIO_mac_cpu_v.1.5.0.3a.zip](https://drive.google.com/file/d/1u9sc0so1w5kkf_g8UBTVJ61gAQ84W2oF/view?usp=sharing) 509MB
- [MMVCServerSIO_win_cuda_v.1.5.0.1a.zip](https://drive.google.com/file/d/1Am0awpHS7NAotITdbnEPg4gl4lxvgWhX/view?usp=sharing) 2.5GB
- [MMVCServerSIO_mac_cpu_v.1.5.0.1a.zip](https://drive.google.com/file/d/19yREPKWe88ycAFPXCPVVYnUatlYfYjTS/view?usp=sharing) 449MB

### 過去バージョン(for v.1.3.x)
- [MMVCServerSIO_mac_onnxcpu_v.1.3.7.2.zip](https://drive.google.com/file/d/1AcJaQXH8ZtlCSrifvRBWdat19HD_A2fr/view?usp=sharing) 365MB
- [MMVCServerSIO_win_onnxdirectML_cuda_v.1.3.7.2.zip](https://drive.google.com/file/d/1WKW3uqmIi9D13Jzao8jWVqx2KANmmQji/view?usp=sharing) 2050MB
- [MMVCServerSIO_win_onnxdirectML_nocuda_v.1.3.7.2.zip](https://drive.google.com/file/d/1b8Lqwb7emvd85NwRANPglKWzceJYcgBg/view?usp=sharing) 286MB
- [MMVCServerSIO_win_onnxgpu_cuda_v.1.3.7.2.zip](https://drive.google.com/file/d/1XcEslgeyo_SsjeFozyWl0zc4izFVGXHE/view?usp=sharing) 2144MB
- [MMVCServerSIO_win_onnxgpu_nocuda_v.1.3.7.2.zip](https://drive.google.com/file/d/1ST7g6jCNm_xe_Q6bj_O-50RMFTDNt_u0/view?usp=sharing) 380MB

- [MMVCServerSIO_mac_onnxcpu_v.1.3.7.0.zip](https://drive.google.com/file/d/1K_ihZ8hxbQq10qrxM1WUfUaj_vY6zwrW/view?usp=sharing) 154MB
- [MMVCServerSIO_win_onnxdirectML_cuda_v.1.3.7.0.zip](https://drive.google.com/file/d/1IJHazaV60ophM6fbmzugZEjulLpBVJUi/view?usp=sharing) 1962MB
- [MMVCServerSIO_win_onnxdirectML_nocuda_v.1.3.7.0.zip](https://drive.google.com/file/d/1_VzdUpiWb8lbIKNppwsFM5pYCAnixOap/view?usp=sharing) 198MB
- [MMVCServerSIO_win_onnxgpu_cuda_v.1.3.7.0.zip](https://drive.google.com/file/d/1uRZHnDq2nVx4oRlXXiqZeE-ZjJlAFx5C/view?usp=sharing) 2057MB
- [MMVCServerSIO_win_onnxgpu_nocuda_v.1.3.7.0.zip](https://drive.google.com/file/d/1DjSCsc_jKaH-TY6qqFbXz7Ya6tS58odb/view?usp=sharing) 293MB
