[Japanese](/README.md) /
[English](/docs_i18n/README_en.md) /
[Korean](/docs_i18n/README_ko.md)/
[Chinese](/docs_i18n/README_zh.md)/
[German](/docs_i18n/README_de.md)/
[Arabic](/docs_i18n/README_ar.md)/
[Greek](/docs_i18n/README_el.md)/
[Spanish](/docs_i18n/README_es.md)/
[French](/docs_i18n/README_fr.md)/
[Italian](/docs_i18n/README_it.md)/
[Latin](/docs_i18n/README_la.md)/
[Malay](/docs_i18n/README_ms.md)/
[Russian](/docs_i18n/README_ru.md)
*Languages other than Japanese are machine translated.

## VCClient

VCClient is software that performs real-time voice conversion using AI.

## What's New!

* v.2.0.72-beta
  * new feature
    * Beatrice Editing GUI
    * Stores pitch and formant for each Beatrice voice
    * GUI Multilingual Support
    * Support for Applio embedder
* v.2.0.70-beta (only for m1 mac)
  * new feature:
    * Supported Beatrice v2 beta.1 on the M1 Mac version of VCClient.
* v.2.0.69-beta (only for win)
  * bugfix:
    * Fixed a bug where the start button would not appear when certain exceptions occurred.
    * Adjusted output buffer for server device mode.
    * Fixed a bug where the sampling rate would change when settings were changed while using server device mode.
    * Fixed bug when using Japanese hubert.
  * misc:
    * Added host API filter for server device mode (highlighted)

## Download and Related Links

Windows and M1 Mac versions can be downloaded from the hugging face repository.

* [VCClient Repository](https://huggingface.co/wok000/vcclient000/tree/main)
* [Light VCClient for Beatrice v2 Repository](https://huggingface.co/wok000/light_vcclient_beatrice/tree/main)

*1 Please clone the repository for Linux use.

### Related Links

* [Beatrice V2 Training Code Repository](https://huggingface.co/fierce-cats/beatrice-trainer)
* [Beatrice V2 Training Code Colab Version](https://github.com/w-okada/beatrice-trainer-colab)

### Related Software

* [Real-time Voice Changer VCClient](https://github.com/w-okada/voice-changer)
* [Text-to-Speech Software TTSClient](https://github.com/w-okada/ttsclient)
* [Real-time Speech Recognition Software ASRClient](https://github.com/w-okada/asrclient)

## Features of VC Client

## Supports various AI models

| AI Model                                                                                                     | v.2       | v.1                  | License                                                                                 |
| ------------------------------------------------------------------------------------------------------------ | --------- | -------------------- | ------------------------------------------------------------------------------------------ |
| [RVC ](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/jp/README.ja.md) | supported | supported            | Please refer to the repository.                                                             |
| [Beatrice v1](https://prj-beatrice.com/)                                                                     | n/a       | supported (only win) | [Proprietary](https://github.com/w-okada/voice-changer/tree/master/server/voice_changer/Beatrice) |
| [Beatrice v2](https://prj-beatrice.com/)                                                                     | supported | n/a                  | [Proprietary](https://huggingface.co/wok000/vcclient_model/blob/main/beatrice_v2_beta/readme.md)  |
| [MMVC](https://github.com/isletennos/MMVC_Trainer)                                                           | n/a       | supported            | Please refer to the repository.                                                             |
| [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)                                               | n/a       | supported            | Please refer to the repository.                                                             |
| [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)                                                               | n/a       | supported            | Please refer to the repository.                                                             |

## Supports both standalone and network configurations

Supports voice conversion completed on a local PC as well as voice conversion via network.
By using it over a network, you can offload the voice conversion load externally when using it simultaneously with high-load applications such as games.

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

## Compatible with multiple platforms

Windows, Mac(M1), Linux, Google Colab

*1 Please clone the repository for Linux use.

## Provides REST API

Clients can be created in various programming languages.

You can also operate it using HTTP clients built into the OS, such as curl.

## Troubleshoot

[Communication Edition](tutorials/trouble_shoot_communication_ja.md)

## About Developer Signature

This software is not signed by the developer. A warning will appear as shown below, but you can run it by clicking the icon while holding down the control key. This is due to Apple's security policy. Execution is at your own risk.

![image](https://user-images.githubusercontent.com/48346627/212567711-c4a8d599-e24c-4fa3-8145-a5df7211f023.png)

## Acknowledgments

* [Tachizundamon Materials](https://seiga.nicovideo.jp/seiga/im10792934)
* [Irasutoya](https://www.irasutoya.com/)
* [Tsukuyomi-chan](https://tyc.rei-yumesaki.net/)

```
  本ソフトウェアの音声合成には、フリー素材キャラクター「つくよみちゃん」が無料公開している音声データを使用しています。
  ■つくよみちゃんコーパス（CV.夢前黎）
  https://tyc.rei-yumesaki.net/material/corpus/
  © Rei Yumesaki
```

* [Amitaro's Voice Material Workshop](https://amitaro.net/)
* [Replica Doll](https://kikyohiroto1227.wixsite.com/kikoto-utau)

## Terms of Use

* Regarding the real-time voice changer Tsukuyomi-chan, it is prohibited to use the converted voice for the following purposes in accordance with the terms of use of the Tsukuyomi-chan corpus.

```

■人を批判・攻撃すること。（「批判・攻撃」の定義は、つくよみちゃんキャラクターライセンスに準じます）

■特定の政治的立場・宗教・思想への賛同または反対を呼びかけること。

■刺激の強い表現をゾーニングなしで公開すること。

■他者に対して二次利用（素材としての利用）を許可する形で公開すること。
※鑑賞用の作品として配布・販売していただくことは問題ございません。
```

* Regarding the real-time voice changer Amitaro, it complies with the following terms of use of Amitaro's Voice Material Workshop. For details,[here](https://amitaro.net/voice/faq/#index_id6)

```
あみたろの声素材やコーパス読み上げ音声を使って音声モデルを作ったり、ボイスチェンジャーや声質変換などを使用して、自分の声をあみたろの声に変換して使うのもOKです。

ただしその場合は絶対に、あみたろ（もしくは小春音アミ）の声に声質変換していることを明記し、あみたろ（および小春音アミ）が話しているわけではないことが誰でもわかるようにしてください。
また、あみたろの声で話す内容は声素材の利用規約の範囲内のみとし、センシティブな発言などはしないでください。
```

* Regarding the real-time voice changer Koto Mahiro, it complies with the terms of use of Replica Doll. For details,[here](https://kikyohiroto1227.wixsite.com/kikoto-utau/ter%EF%BD%8Ds-of-service)

## Disclaimer

We are not responsible for any direct, indirect, consequential, or special damages arising from the use or inability to use this software.
