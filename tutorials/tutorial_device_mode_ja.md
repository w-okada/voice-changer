## Tutorial Device Mode

デバイスモードについて説明します。

[説明動画](https://youtu.be/SUnRGCJ92K8?t=99)

## v.1.5.2.9 以前の構成(client device mode)

v.1.5.2.9 以前はブラウザが制御するマイクとスピーカを用いてボイチェンを行っていました。
これを client device モードと呼びます(赤線)。

![image](https://github.com/w-okada/voice-changer/assets/48346627/56c0766c-45c1-4b3d-af66-73443c232807)

## v.1.5.2.9 以降の構成(client device mode / server device mode)

v.1.5.2.9 より PC に接続されたマイクとスピーカーを直接 VC Client から制御してボイチェンを行えるモードを追加しました。これを server device mode と呼びます(青線)。

![image](https://github.com/w-okada/voice-changer/assets/48346627/34c92e36-0662-4eeb-aac5-30cd1f4a5cd8)

## client device mode / server device mode のメリットとデメリット

v.1.5.2.9 以降では、client device mode と server device mode のどちらを使うかを選択できます。

- client device mode
  - good points
    1. Chrome がマイク/スピーカーの難しい処理を請け負ってくれる。
    2. ノイズ除去などの Chrome が持つ Web 会議向け機能が使える
  - bad points
    1. 多少遅延が増える
- server device mode
  - good points
    1. VC Client が直接マイク/スピーカーを扱うので遅延が少ない。
  - bad points
    1. 扱えないマイク/スピーカーがあるかも。
    2. ノイズ除去など Chrome の便利機能が使えない。

![image](https://github.com/w-okada/voice-changer/assets/48346627/fef1ee63-e853-4867-b4c8-bf0121495bb6)

ユーザはそれぞれのメリット・デメリットを考慮して使い分けることができます。
