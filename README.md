## VC Client

[English](/README_en.md) [Korean](/README_ko.md)

## What's New!
- Beatrice V2 トレーニングコード公開!!!
  - [トレーニングコードリポジトリ](https://huggingface.co/fierce-cats/beatrice-trainer)
  - [コラボ版](https://github.com/w-okada/beatrice-trainer-colab)
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
- v.2.0.65-beta
  - new feature: Beatrice v2 beta.1をサポートしました。さらなる高品質な音声変換が可能になります。
- v.2.0.61-alpha
  - feature:
    - クロスフェードの時間を指定できるようになりました。
  - bugfix:
    - モデルマージの際に、使用しないモデルの要素を0にしても動くようになりました。
- v.2.0.58-alpha
  - feature:
    - SIO ブロードキャスティング
    - embed ngrok(experimental)
  - improve:
    - for Mobile Phone tuning.
  - bugfix:
    - macos CUIメッセージ文字化け
- v.2.0.55-alpha
  - improve:
    - RVCのCPU負荷を削減
    - WebSocket対応
  - change
    - 起動バッチでno_cuiオプションを有効化

# VC Client とは

1. 各種音声変換 AI(VC, Voice Conversion)を用いてリアルタイム音声変換を行うためのクライアントソフトウェアです。サポートしている音声変換 AI は次のものになります。

- サポートする音声変換 AI （サポート VC）
  - [RVC(Retrieval-based-Voice-Conversion)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI)
  - [Beatrice JVS Corpus Edition](https://prj-beatrice.com/) * experimental,  (***NOT MIT Licnsence*** see [readme](https://github.com/w-okada/voice-changer/blob/master/server/voice_changer/Beatrice/)) *  Only for Windows, CPU dependent
1. 本ソフトウェアは、ネットワークを介した利用も可能であり、ゲームなどの高負荷なアプリケーションと同時に使用する場合などに音声変換処理の負荷を外部にオフロードすることができます。

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

1. 複数のプラットフォームに対応しています。

- Windows, Mac(M1), Linux, Google Colab (MMVC のみ)

1. REST APIを提供しています。

- curlなどのOSに組み込まれているHTTPクライアントを使って操作ができます。
- これにより、次のようなことが簡単に実現できます。
  - ユーザが.bat等でREST APIをたたく処理をショートカットとして登録する。
  - リモートから操作する簡易クライアントを作成する。
  - など。
  - 
## 関連ソフトウェア
- [リアルタイムボイスチェンジャ VCClient](https://github.com/w-okada/voice-changer)
- [読み上げソフトウェア TTSClient](https://github.com/w-okada/ttsclient)
- [リアルタイム音声認識ソフトウェア ASRClient](https://github.com/w-okada/asrclient)

# ダウンロード
[Hugging Face](https://huggingface.co/wok000/vcclient000/tree/main)からダウンロードしてください。

# マニュアル

[マニュアル](docs/01_basic_v2.0.z.md)


# トラブルシュート

- [通信編](tutorials/trouble_shoot_communication_ja.md)


# 開発者の署名について

本ソフトウェアは開発元の署名しておりません。下記のように警告が出ますが、コントロールキーを押しながらアイコンをクリックすると実行できるようになります。これは Apple のセキュリティポリシーによるものです。実行は自己責任となります。

![image](https://user-images.githubusercontent.com/48346627/212567711-c4a8d599-e24c-4fa3-8145-a5df7211f023.png)

# Acknowledgments

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

# 利用規約

- リアルタイムボイスチェンジャーつくよみちゃんについては、つくよみちゃんコーパスの利用規約に準じ、次の目的で変換後の音声を使用することを禁止します。

```

■人を批判・攻撃すること。（「批判・攻撃」の定義は、つくよみちゃんキャラクターライセンスに準じます）

■特定の政治的立場・宗教・思想への賛同または反対を呼びかけること。

■刺激の強い表現をゾーニングなしで公開すること。

■他者に対して二次利用（素材としての利用）を許可する形で公開すること。
※鑑賞用の作品として配布・販売していただくことは問題ございません。
```

- リアルタイムボイスチェンジャーあみたろについては、あみたろの声素材工房様の次の利用規約に準じます。詳細は[こちら](https://amitaro.net/voice/faq/#index_id6)です。

```
あみたろの声素材やコーパス読み上げ音声を使って音声モデルを作ったり、ボイスチェンジャーや声質変換などを使用して、自分の声をあみたろの声に変換して使うのもOKです。

ただしその場合は絶対に、あみたろ（もしくは小春音アミ）の声に声質変換していることを明記し、あみたろ（および小春音アミ）が話しているわけではないことが誰でもわかるようにしてください。
また、あみたろの声で話す内容は声素材の利用規約の範囲内のみとし、センシティブな発言などはしないでください。
```

- リアルタイムボイスチェンジャー黄琴まひろについては、れぷりかどーるの利用規約に準じます。詳細は[こちら](https://kikyohiroto1227.wixsite.com/kikoto-utau/ter%EF%BD%8Ds-of-service)です。

# 免責事項

本ソフトウェアの使用または使用不能により生じたいかなる直接損害・間接損害・波及的損害・結果的損害 または特別損害についても、一切責任を負いません。
