[日本語](/README.md) /
[英語](/docs_i18n/README_en.md) /
[韓国語](/docs_i18n/README_ko.md)/
[中国語](/docs_i18n/README_zh.md)/
[ドイツ語](/docs_i18n/README_de.md)/
[アラビア語](/docs_i18n/README_ar.md)/
[ギリシャ語](/docs_i18n/README_el.md)/
[スペイン語](/docs_i18n/README_es.md)/
[フランス語](/docs_i18n/README_fr.md)/
[イタリア語](/docs_i18n/README_it.md)/
[ラテン語](/docs_i18n/README_la.md)/
[マレー語](/docs_i18n/README_ms.md)/
[ロシア語](/docs_i18n/README_ru.md) \*日本語以外は機械翻訳です。

## VCClient

VCClient は、AI を用いてリアルタイム音声変換を行うソフトウェアです。

## What's New!
- v.2.1.3-alpha

  - ショートカットキー
  - バッファの可視化
  - currently only for rvc

- v.2.0.78-beta

  - bugfix: RVC モデルのアップロードエラーを回避
  - ver.1.x との同時起動ができるようになりました。
  - 選択できる chunk size を増やしました。

- v.2.0.77-beta (only for RTX 5090, experimental)
  - 関連モジュールを 5090 対応 (開発者が RTX5090 未所持のため、動作未検証)
- v.2.0.76-beta
  - new feature:
    - Beatrice: 話者マージの実装
    - Beatrice: オートピッチシフト
  - bugfix:
    - サーバモードのデバイス選択時の不具合対応
- v.2.0.73-beta
  - new feature:
    - 編集した beatrice model のダウンロード
  - bugfix:
    - beatrice v2 の pitch, formant が反映されないバグを修正
    - Applio の embedder を使用しているモデルの ONNX ができないバグを修正

## エディション
v2.2.1以降は、エディションによりサポートする AI モデルが異なります。

| edition | Support Model |     |
| ------- | ------------- | --- |
| std     | Beatrice      |     |
| cuda    | Beatrice, RVC |     |
| onnx    | Beatrice, RVC |     |

## ダウンロードと関連リンク

Windows 版、 M1 Mac 版は hugging face のリポジトリからダウンロードできます。

- [VCClient のリポジトリ](https://huggingface.co/wok000/vcclient000/tree/main)
- [Light VCClient for Beatrice v2 のリポジトリ](https://huggingface.co/wok000/light_vcclient_beatrice/tree/main)

\*1 Linux はリポジトリを clone してお使いください。

### 関連リンク

- [Beatrice V2 トレーニングコードのリポジトリ](https://huggingface.co/fierce-cats/beatrice-trainer)
- [Beatrice V2 トレーニングコード Colab 版](https://github.com/w-okada/beatrice-trainer-colab)

### 関連ソフトウェア

- [リアルタイムボイスチェンジャ VCClient](https://github.com/w-okada/voice-changer)
- [読み上げソフトウェア TTSClient](https://github.com/w-okada/ttsclient)
- [リアルタイム音声認識ソフトウェア ASRClient](https://github.com/w-okada/asrclient)

## VC Client の特徴

## 多様な AI モデルをサポート

| AI モデル                                                                                                    | v.2       | v.1                  | ライセンス                                                                                 |
| ------------------------------------------------------------------------------------------------------------ | --------- | -------------------- | ------------------------------------------------------------------------------------------ |
| [RVC ](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/jp/README.ja.md) | supported | supported            | リポジトリを参照してください。                                                             |
| [Beatrice v1](https://prj-beatrice.com/)                                                                     | n/a       | supported (only win) | [独自](https://github.com/w-okada/voice-changer/tree/master/server/voice_changer/Beatrice) |
| [Beatrice v2](https://prj-beatrice.com/)                                                                     | supported | n/a                  | [独自](https://huggingface.co/wok000/vcclient_model/blob/main/beatrice_v2_beta/readme.md)  |
| [MMVC](https://github.com/isletennos/MMVC_Trainer)                                                           | n/a       | supported            | リポジトリを参照してください。                                                             |
| [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)                                               | n/a       | supported            | リポジトリを参照してください。                                                             |
| [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)                                                               | n/a       | supported            | リポジトリを参照してください。                                                             |

## スタンドアロン、ネットワーク経由の両構成をサポート

ローカル PC で完結した音声変換も、ネットワークを介した音声変換もサポートしています。
ネットワークを介した利用を行うことで、ゲームなどの高負荷なアプリケーションと同時に使用する場合に音声変換の負荷を外部にオフロードすることができます。

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

## 複数プラットフォームに対応

Windows, Mac(M1), Linux, Google Colab

\*1 Linux はリポジトリを clone してお使いください。

## REST API を提供

各種プログラミング言語でクライアントを作成することができます。

また、curl などの OS に組み込まれている HTTP クライアントを使って操作ができます。

## トラブルシュート

[通信編](tutorials/trouble_shoot_communication_ja.md)

## 開発者の署名について

本ソフトウェアは開発元の署名しておりません。下記のように警告が出ますが、コントロールキーを押しながらアイコンをクリックすると実行できるようになります。これは Apple のセキュリティポリシーによるものです。実行は自己責任となります。

![image](https://user-images.githubusercontent.com/48346627/212567711-c4a8d599-e24c-4fa3-8145-a5df7211f023.png)

## Acknowledgments

- [立ちずんだもん素材](https://seiga.nicovideo.jp/seiga/im10792934)
- [いらすとや](https://www.irasutoya.com/)
- [つくよみちゃん](https://tyc.rei-yumesaki.net/)

```
  本ソフトウェアの音声合成には、フリー素材キャラクター「つくよみちゃん」が無料公開している音声データを使用しています。
  ■つくよみちゃんコーパス（CV.夢前黎）
  https://tyc.rei-yumesaki.net/material/corpus/
  © Rei Yumesaki
```

- [あみたろの声素材工房](https://amitaro.net/)
- [れぷりかどーる](https://kikyohiroto1227.wixsite.com/kikoto-utau)

## 利用規約

- リアルタイムボイスチェンジャーつくよみちゃんについては、つくよみちゃんコーパスの利用規約に準じ、次の目的で変換後の音声を使用することを禁止します。

```

■人を批判・攻撃すること。（「批判・攻撃」の定義は、つくよみちゃんキャラクターライセンスに準じます）

■特定の政治的立場・宗教・思想への賛同または反対を呼びかけること。

■刺激の強い表現をゾーニングなしで公開すること。

■他者に対して二次利用（素材としての利用）を許可する形で公開すること。
※鑑賞用の作品として配布・販売していただくことは問題ございません。
```

- リアルタイムボイスチェンジャーあみたろについては、あみたろの声素材工房様の次の利用規約に準じます。詳細は[こちら](https://amitaro.net/voice/faq/#index_id6)

```
あみたろの声素材やコーパス読み上げ音声を使って音声モデルを作ったり、ボイスチェンジャーや声質変換などを使用して、自分の声をあみたろの声に変換して使うのもOKです。

ただしその場合は絶対に、あみたろ（もしくは小春音アミ）の声に声質変換していることを明記し、あみたろ（および小春音アミ）が話しているわけではないことが誰でもわかるようにしてください。
また、あみたろの声で話す内容は声素材の利用規約の範囲内のみとし、センシティブな発言などはしないでください。
```

- リアルタイムボイスチェンジャー黄琴まひろについては、れぷりかどーるの利用規約に準じます。詳細は[こちら](https://kikyohiroto1227.wixsite.com/kikoto-utau/ter%EF%BD%8Ds-of-service)

## 免責事項

本ソフトウェアの使用または使用不能により生じたいかなる直接損害・間接損害・波及的損害・結果的損害 または特別損害についても、一切責任を負いません。
