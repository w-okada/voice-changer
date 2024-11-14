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
[ロシア語](/docs_i18n/README_ru.md) 
  *日本語以外は機械翻訳です。

VCClient
---

VCClientは、AIを用いてリアルタイム音声変換を行うソフトウェアです。

## What's New!
- v.2.0.73-beta
  - new feature:
    - 編集したbeatrice modelのダウンロード
  - bugfix:
    - beatrice v2 のpitch, formantが反映されないバグを修正
    - Applio のembedderを使用しているモデルのONNXができないバグを修正
- v.2.0.72-beta (バグがあるので非推奨。v.2.0.73で修正済み)
  - new feature
    - Beatriceの編集GUI
    - Beatriceのvoiceごとにpitch, formantを記憶
    - GUI多言語化
    - Applioのembedder対応
- v.2.0.70-beta (only for m1 mac)
  - new feature:
    - M1 Mac版VCClientでもBeatrice v2 beta.1をサポートしました。
- v.2.0.69-beta (only for win)
  - bugfix:
    - 一部の例外発生時にスタートボタンが表示されなくなるバグを修正
    - サーバデバイスモードの出力バッファを調整
    - サーバデバイスモード使用中に設定変更を行うとサンプリングレートが変化するバグを修正
    - 日本語hubert使用時のバグ修正
  - misc:
    - サーバデバイスモードのホストAPIフィルタ追加（強調表示）



## ダウンロードと関連リンク
Windows版、 M1 Mac版はhugging faceのリポジトリからダウンロードできます。

- [VCClient のリポジトリ](https://huggingface.co/wok000/vcclient000/tree/main)
- [Light VCClient for Beatrice v2 のリポジトリ](https://huggingface.co/wok000/light_vcclient_beatrice/tree/main)


*1 Linuxはリポジトリをcloneしてお使いください。

### 関連リンク
- [Beatrice V2 トレーニングコードのリポジトリ](https://huggingface.co/fierce-cats/beatrice-trainer)
- [Beatrice V2 トレーニングコード Colab版](https://github.com/w-okada/beatrice-trainer-colab)

### 関連ソフトウェア
- [リアルタイムボイスチェンジャ VCClient](https://github.com/w-okada/voice-changer)
- [読み上げソフトウェア TTSClient](https://github.com/w-okada/ttsclient)
- [リアルタイム音声認識ソフトウェア ASRClient](https://github.com/w-okada/asrclient)



## VC Clientの特徴

## 多様なAIモデルをサポート

| AIモデル                                                                                                     | v.2       | v.1                  | ライセンス                                                                                 |
| ------------------------------------------------------------------------------------------------------------ | --------- | -------------------- | ------------------------------------------------------------------------------------------ |
| [RVC ](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/jp/README.ja.md) | supported | supported            | リポジトリを参照してください。                                                             |
| [Beatrice v1](https://prj-beatrice.com/)                                                                     | n/a       | supported (only win) | [独自](https://github.com/w-okada/voice-changer/tree/master/server/voice_changer/Beatrice) |
| [Beatrice v2](https://prj-beatrice.com/)                                                                     | supported | n/a                  | [独自](https://huggingface.co/wok000/vcclient_model/blob/main/beatrice_v2_beta/readme.md)  |
| [MMVC](https://github.com/isletennos/MMVC_Trainer)                                                           | n/a       | supported            | リポジトリを参照してください。                                                             |
| [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)                                               | n/a       | supported            | リポジトリを参照してください。                                                             |
| [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)                                                               | n/a       | supported            | リポジトリを参照してください。                                                             |

## スタンドアロン、ネットワーク経由の両構成をサポート
ローカルPCで完結した音声変換も、ネットワークを介した音声変換もサポートしています。
ネットワークを介した利用を行うことで、ゲームなどの高負荷なアプリケーションと同時に使用する場合に音声変換の負荷を外部にオフロードすることができます。

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

## 複数プラットフォームに対応

Windows, Mac(M1), Linux, Google Colab

*1 Linuxはリポジトリをcloneしてお使いください。

## REST APIを提供
各種プログラミング言語でクライアントを作成することができます。

また、curlなどのOSに組み込まれているHTTPクライアントを使って操作ができます。

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
