[日语](/README.md) /
[英语](/docs_i18n/README_en.md) /
[韩语](/docs_i18n/README_ko.md)/
[中文](/docs_i18n/README_zh.md)/
[德语](/docs_i18n/README_de.md)/
[阿拉伯语](/docs_i18n/README_ar.md)/
[希腊语](/docs_i18n/README_el.md)/
[西班牙语](/docs_i18n/README_es.md)/
[法语](/docs_i18n/README_fr.md)/
[意大利语](/docs_i18n/README_it.md)/
[拉丁语](/docs_i18n/README_la.md)/
[马来语](/docs_i18n/README_ms.md)/
[俄语](/docs_i18n/README_ru.md)
*除日语外，其他语言均为机器翻译。

## VCClient

VCClient是一款利用AI进行实时语音转换的软件。

## What's New!

* v.2.0.77-beta (仅适用于 RTX 5090，实验性)
  * 相关模块支持 RTX 5090（由于开发者未拥有 RTX 5090，未经验证）
* v.2.0.76-beta
  * 新功能：
    * Beatrice: 实现说话者合并
    * Beatrice: 自动音高转换
  * 错误修复：
    * 修复服务器模式下设备选择的问题
* v.2.0.73-beta
  * 新功能：
    * 下载编辑后的beatrice模型
  * 错误修复：
    * 修复了beatrice v2的音高和共振峰未反映的错误
    * 修复了使用Applio的embedder的模型无法生成ONNX的错误

## 下载和相关链接

Windows版、M1 Mac版可以从hugging face的仓库下载。

* [VCClient 的仓库](https://huggingface.co/wok000/vcclient000/tree/main)
* [Light VCClient for Beatrice v2 的仓库](https://huggingface.co/wok000/light_vcclient_beatrice/tree/main)

*1 Linux请克隆仓库使用。

### 相关链接

* [Beatrice V2 训练代码的仓库](https://huggingface.co/fierce-cats/beatrice-trainer)
* [Beatrice V2 训练代码 Colab版](https://github.com/w-okada/beatrice-trainer-colab)

### 相关软件

* [实时变声器 VCClient](https://github.com/w-okada/voice-changer)
* [语音合成软件 TTSClient](https://github.com/w-okada/ttsclient)
* [实时语音识别软件 ASRClient](https://github.com/w-okada/asrclient)

## VC Client的特点

## 支持多种AI模型

| AI模型                                                                                                     | v.2       | v.1                  | 许可证                                                                                 |
| ------------------------------------------------------------------------------------------------------------ | --------- | -------------------- | ------------------------------------------------------------------------------------------ |
| [RVC ](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/jp/README.ja.md) | supported | supported            | 请参阅仓库。                                                             |
| [Beatrice v1](https://prj-beatrice.com/)                                                                     | n/a       | supported (only win) | [独立](https://github.com/w-okada/voice-changer/tree/master/server/voice_changer/Beatrice) |
| [Beatrice v2](https://prj-beatrice.com/)                                                                     | supported | n/a                  | [独立](https://huggingface.co/wok000/vcclient_model/blob/main/beatrice_v2_beta/readme.md)  |
| [MMVC](https://github.com/isletennos/MMVC_Trainer)                                                           | n/a       | supported            | 请参阅仓库。                                                             |
| [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)                                               | n/a       | supported            | 请参阅仓库。                                                             |
| [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)                                                               | n/a       | supported            | 请参阅仓库。                                                             |

## 支持独立和通过网络的两种配置

支持在本地PC上完成的语音转换和通过网络的语音转换。
通过网络使用时，可以在与游戏等高负荷应用程序同时使用时将语音转换的负荷转移到外部。

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

## 支持多平台

Windows, Mac(M1), Linux, Google Colab

*1 Linux请克隆仓库使用。

## 提供REST API

可以用各种编程语言创建客户端。

还可以使用curl等操作系统内置的HTTP客户端进行操作。

## 故障排除

[通信篇](tutorials/trouble_shoot_communication_ja.md)

## 关于开发者的签名

本软件未由开发者签名。虽然会出现如下警告，但按住Control键并点击图标即可运行。这是由于Apple的安全策略所致。运行需自行承担风险。

![image](https://user-images.githubusercontent.com/48346627/212567711-c4a8d599-e24c-4fa3-8145-a5df7211f023.png)

## 致谢

* [立ちずんだもん素材](https://seiga.nicovideo.jp/seiga/im10792934)
* [いらすとや](https://www.irasutoya.com/)
* [つくよみちゃん](https://tyc.rei-yumesaki.net/)

```
  本ソフトウェアの音声合成には、フリー素材キャラクター「つくよみちゃん」が無料公開している音声データを使用しています。
  ■つくよみちゃんコーパス（CV.夢前黎）
  https://tyc.rei-yumesaki.net/material/corpus/
  © Rei Yumesaki
```

* [あみたろの声素材工房](https://amitaro.net/)
* [れぷりかどーる](https://kikyohiroto1227.wixsite.com/kikoto-utau)

## 使用条款

* 关于实时变声器つくよみちゃん，禁止将转换后的语音用于以下目的，遵循つくよみちゃん语料库的使用条款。

```

■人を批判・攻撃すること。（「批判・攻撃」の定義は、つくよみちゃんキャラクターライセンスに準じます）

■特定の政治的立場・宗教・思想への賛同または反対を呼びかけること。

■刺激の強い表現をゾーニングなしで公開すること。

■他者に対して二次利用（素材としての利用）を許可する形で公開すること。
※鑑賞用の作品として配布・販売していただくことは問題ございません。
```

* 关于实时变声器あみたろ，遵循あみたろの声素材工房的以下使用条款。详情请见[这里](https://amitaro.net/voice/faq/#index_id6)

```
あみたろの声素材やコーパス読み上げ音声を使って音声モデルを作ったり、ボイスチェンジャーや声質変換などを使用して、自分の声をあみたろの声に変換して使うのもOKです。

ただしその場合は絶対に、あみたろ（もしくは小春音アミ）の声に声質変換していることを明記し、あみたろ（および小春音アミ）が話しているわけではないことが誰でもわかるようにしてください。
また、あみたろの声で話す内容は声素材の利用規約の範囲内のみとし、センシティブな発言などはしないでください。
```

* 关于实时变声器黄琴まひろ，遵循れぷりかどーる的使用条款。详情请见[这里](https://kikyohiroto1227.wixsite.com/kikoto-utau/ter%EF%BD%8Ds-of-service)

## 免责声明

对于因使用或无法使用本软件而导致的任何直接、间接、衍生、结果性或特殊损害，本软件概不负责。
