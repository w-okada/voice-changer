## Tutorial Monitor Output

v.1.5.3.7 から追加された server device mode における monitor output について説明します。

## v.1.5.3.6 以前の構成

出力先デバイスが一つしか設定できませんでした。Discord や Zoom など他のアプリケーションと連携するためには、一般的にこの出力先を Voicemeeter などの仮想オーディオデバイスに設定する必要があります。このため、変換後の音声を確認するためには仮想オーディオデバイスを経由して行う必要があり、多くのオーバーヘッドがかかっていました(青線)。

![image](https://github.com/w-okada/voice-changer/assets/48346627/faba8fdf-cfa5-468f-a56b-3fa986fb45a1)

## v.1.5.3.7 以降の構成

v.1.5.3.7 では、VCClient の server device mode でもう一つ出力先デバイスを設定できるようになりました(赤線)。これにより、モニター用には Voicemeeter を経由せずに直接 wasapi デバイスや asio デバイスに出力できるようになり、遅延が少ないモニタリングが可能になります。

![image](https://github.com/w-okada/voice-changer/assets/48346627/1d5065eb-b042-4521-ade3-66828c87a712)

## 使い方

デバイス設定エリアで server device mode を選択してください。サンプリングレート(S.R.)、input, output, monitor を設定できるようになります。

![image](https://github.com/w-okada/voice-changer/assets/48346627/c15e6800-75ec-410b-87f2-c96d0c697c91)

## 注意事項

server device mode で使用する input, output, monitor のそれぞれのデバイスはサンプリングレートが一致している必要があります。一致していない場合は、コンソール上に詳細情報がでるのでそれぞれのデバイスが対応しているサンプリングレートを GUI から指定してください。

### 例

![image](https://github.com/w-okada/voice-changer/assets/48346627/d621d356-5710-4766-932e-43b7d520df5f)

サンプリングレートが一致していないとこのような表示がでます。

(1)は現在 GUI で指定したサンプリングレートにデバイスが対応しているかを表示しています。False のデバイスは対応していません。

(2)で各デバイスで対応可能なサンプリングレートが表示されます。input, output, monitor のすべてで対応可能なサンプリングレートを指定してください。ここでは 48000 を指定することになります。

## Tips

### その１

お使いの環境により大きく変わると思いますが、開発者の環境では Input, Monitor を Wasapi デバイス、output を任意にすることで遅延をかなり少なく運用することができました。
(RTX4090 使用)

### その 2

Wasapi のサンプリングレートはデバイス側で設定したものしか選択できません。この設定は Windows のサウンド設定から変更できます。(Win11)

![image](https://github.com/w-okada/voice-changer/assets/48346627/300c8cf0-cb7d-4f24-8253-fa313caee5df)
