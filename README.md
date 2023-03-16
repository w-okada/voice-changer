VC Helper
----
# VC Helperとは
[VC Helper](https://github.com/w-okada/voice-changer)は[MMVC](https://github.com/isletennos/MMVC_Trainer), [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)などのAIを使ったリアルタイムボイスチェンジャーのヘルパーアプリケーションです。リアルタイムボイスチェンジャーで必要となるトレーニング用の音声の録音(MMVC向け)とボイスチェンジャを各種プラットフォームでお手軽に実行できます。

[解説動画](https://www.nicovideo.jp/watch/sm41507891)

※ MMVCのトレーニングについては[公式ノートブック](https://github.com/isletennos/MMVC_Trainer)をご利用ください。
※ so-vits-svcのトレーニングについては[公式](https://github.com/svc-develop-team/so-vits-svc)を参考に実施してください。

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
MMVCやso-vits-svcでボイチェンを行うためのアプリです。

大きく3つの方法でご利用できます。難易度順に次の通りです。
- Google Colaboratoryでの利用(MMVCのみ)
- 事前ビルド済みのBinaryでの利用
- DockerやAnacondaなど環境構築を行った上での利用

本ソフトウェアやMMVCになじみの薄い方は上から徐々に慣れていくとよいと思います。

## (2-1) Google Colaboratoryでの利用(MMVCのみ)
Googleが提供している機械学習プラットフォームColaboratory上で実行できます。
MMVCのモデルをトレーニングが完了している場合、既にColaboratoryを利用していると思いますので、事前準備は必要ありません。ただし、ネットワーク環境やColaboratoryの状況によってボイスチェンジャのタイムラグが大きくなる傾向があります。

- [超簡単バージョン](https://github.com/w-okada/voice-changer/blob/master/VoiceChangerDemo_Simple.ipynb): 事前設定なしでColabから実行できます。 
- [普通バージョン](https://github.com/w-okada/voice-changer/blob/master/VoiceChangerDemo.ipynb): Google Driveと連携してモデルを読み込むことができます。

[解説動画](https://youtu.be/TogfMzXH1T0)

## (2-2) 事前ビルド済みのBinaryでの利用
実行形式のバイナリをダウンロードして実行することができます。
Windows版とMac版を提供しています。

・Mac版はダウンロードファイルを解凍したのちに、使用するVCに応じた`startHttp_xxx.command`をダブルクリックしてください。開発元を検証できない旨が示される場合は、再度コントロールキーを押してクリックして実行してください(or 右クリックから実行してください)。（詳細下記 *1）

・Windows版は、ONNX版とONNX+PyTorch版を提供しています。環境に応じたzipファイルをダウンロードしてください。ダウンロードしたzipファイルを解凍して、使用するVCに応じたVCに応じた`start_http_xxx.bat`を実行してください。

・NvidiaのGPUをお持ちの方は多くの場合はONNX版で動きます。環境によって極まれにgpuが認識されない場合があります。その場合はONNX+PyTorch(cuda)版の方をご利用ください。（サイズが大きく違います。）

・NvidiaのGPUをお持ちでない方は多くの場合はONNX(DirectML)版で動きます。

・リモートからアクセスできるようにする方法など、より詳しくは[こちら](https://zenn.dev/wok/books/0002_vc-helper-v_1_3)をご覧ください。

・so-vits-svcは[4.0-v2](https://github.com/svc-develop-team/so-vits-svc/tree/4.0-v2)に対応しています。

・so-vits-svcやつくよみちゃんの動作にはcontent vecのモデルが必要となります。こちらの[リポジトリ](https://github.com/auspicious3000/contentvec)から、ContentVec_legacy	500のモデルをダウンロードして、実行する`startHttp_xxx.command`や`start_http_xxx.bat`と同じフォルダに配置してください。


| Version            | OS      | フレームワーク               | link                                                                                                 | サポートVC                                    | サイズ |
| ------------------ | ------- | ---------------------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------- | ------ |
| v.1.5.1.6(current) | mac(M1) | ONNX                         | [通常](https://drive.google.com/file/d/1espnNjLnSr4BYWL6ilb34bkvcVAbByTu/view?usp=sharing)           | MMVC v.1.5.x, MMVC v.1.3.x, so-vits-svc 4.0v2 | 571MB  |
|                    |         |                              | [つくよみちゃん](https://drive.google.com/file/d/1iXDowAEKudvIsyeDDrqxEEawm3IJr-Cb/view?usp=sharing) | so-vits-svc 4.0v2                             | 949MB  |
|                    | windows | ONNX                         | [通常](https://drive.google.com/file/d/1rHUCo-URl13TdE3v8Vb_DD4jUG_yGTuu/view?usp=sharing)           | MMVC v.1.5.x, MMVC v.1.3.x                    | 564MB  |
|                    |         | ONNX+PyTorch(cuda)           | [通常](https://drive.google.com/file/d/1K2jkUHEDkLzbYligyVJcuVnOX1KQAQiR/view?usp=sharing)           | MMVC v.1.5.x, MMVC v.1.3.x, so-vits-svc 4.0v2 | 2.6GB  |
|                    |         |                              | [つくよみちゃん](https://drive.google.com/file/d/1hPxPFGNVfrK5E-K18xrSH9gPJRYztG9d/view?usp=sharing) | so-vits-svc 4.0v2                             | 2.97GB |
|                    |         | ONNX(DirectML)               | [通常](https://drive.google.com/file/d/1mjhmB2plYGuw6G4qTT3IoWFDU5e8yOTR/view?usp=sharing)           | MMVC v.1.5.x, MMVC v.1.3.x                    | 452MB  |
|                    |         | ONNX(DirectML)+PyTorch(cuda) | [通常](https://drive.google.com/file/d/1KlNRQaD-hnPTo1e6sZi-9jDOq1TNI4E6/view?usp=sharing)           | MMVC v.1.5.x, MMVC v.1.3.x                    | 2.47GB |

※ MMVC v.1.5.xはExperimentalです。

※ つくよみちゃんはフリー素材キャラクター「つくよみちゃん」が無料公開している音声データを使用しています。（利用規約など、詳細は文末）

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

### 過去バージョン
| Version    | OS      | フレームワーク          | link                                                                                       | サポートVC   | サイズ |
| ---------- | ------- | ----------------------- | ------------------------------------------------------------------------------------------ | ------------ | ------ |
| v.1.5.1.4a | mac(M1) | onnx(cpu)               | [通常](https://drive.google.com/file/d/1urqcB_S4lqbrxL4osKIlQ6MLhsh__W7t/view?usp=sharing) | MMVC v.1.5.x | 510MB  |
|            | windows | onnx(cpu)               | [通常](https://drive.google.com/file/d/1cQVnwenJD0vXzyThZ3iiMBUVLRXoDPBJ/view?usp=sharing) | MMVC v.1.5.x | 541MB  |
|            |         | onnx(cpu+cuda)          | [通常](https://drive.google.com/file/d/1sUa42la2vjTkIMcLKRSBf8icvfB6fqM0/view?usp=sharing) | MMVC v.1.5.x | 2.55GB |
|            |         | onnx(cpu+DirectML)      | [通常](https://drive.google.com/file/d/1B_dPZMIf39Of7olTVzR0h6fNP5u0lx8P/view?usp=sharing) | MMVC v.1.5.x | 430MB  |
|            |         | onnx(cpu+DirectML+cuda) | [通常](https://drive.google.com/file/d/1KSmmu5A29f3wXc_ZreycuuCyLltgNg5h/view?usp=sharing) | MMVC v.1.5.x | 2.45GB |
| ---        | ---     | ---                     | ---                                                                                        | ---          | ---    |
| v.1.3.1.4a | mac(M1) | onnx(cpu)               | [通常](https://drive.google.com/file/d/1dliqQE7Kn5vhycrDUZQ6pgwLfP_znAyp/view?usp=sharing) | MMVC v.1.3.x | 510MB  |
|            | windows | onnx(cpu)               | [通常](https://drive.google.com/file/d/16R44mbi4AlkynVzhROrmA6u9MRSegpwd/view?usp=sharing) | MMVC v.1.3.x | 541MB  |
|            |         | onnx(cpu+cuda)          | [通常](https://drive.google.com/file/d/1wBgjNHf0Kz3BPp-73KCbCEnPTW4FnuEE/view?usp=sharing) | MMVC v.1.3.x | 2.55GB |
|            |         | onnx(cpu+DirectML)      | [通常](https://drive.google.com/file/d/14DaEPJnio-Ne50e2t1wFRgwQYnWwkh6n/view?usp=sharing) | MMVC v.1.3.x | 430MB  |
|            |         | onnx(cpu+DirectML+cuda) | [通常](https://drive.google.com/file/d/1vfZc52f0BVD8nGjsuaAhl0jb4djHBBYe/view?usp=sharing) | MMVC v.1.3.x | 2.45GB |


# Acknowledgments
- 立ちずんだもん素材：https://seiga.nicovideo.jp/seiga/im10792934
- いらすとや：https://www.irasutoya.com/
- つくよみちゃん

```
  本ソフトウェアの音声合成には、フリー素材キャラクター「つくよみちゃん」が無料公開している音声データを使用しています。
  ■つくよみちゃんコーパス（CV.夢前黎）
  https://tyc.rei-yumesaki.net/material/corpus/
  © Rei Yumesaki
```
# 利用規約
```
リアルタイムボイスチェンジャーつくよみちゃんについては、つくよみちゃんコーパスの利用規約に準じ、次の目的での利用を禁止します。

■人を批判・攻撃すること。（「批判・攻撃」の定義は、つくよみちゃんキャラクターライセンスに準じます）

■特定の政治的立場・宗教・思想への賛同または反対を呼びかけること。

■刺激の強い表現をゾーニングなしで公開すること。

■他者に対して二次利用（素材としての利用）を許可する形で公開すること。
※鑑賞用の作品として配布・販売していただくことは問題ございません。
```
# 免責事項
本ソフトウェアの使用または使用不能により生じたいかなる直接損害・間接損害・波及的損害・結果的損害 または特別損害についても、一切責任を負いません。

