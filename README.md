## VC Client

[English](/README_en.md) [Korean](/README_ko.md)

## What's New!
- v.2.0.44-alpha
  - bugfix
    - モデル削除後の不安定動作の改善
- v.2.0.42-alpha
  - Feature
    - Beatrice v2 カスタムモデルのアップロード
  - Improvement
    - RVC音質向上
    - RVC変換速度向上
  - Bugfix
    - パススルー
    - オーディオデバイスリロード
- v.2.0.40-alpha
  - 改善
    - 音量対応
    - ASIO対応
    - webフォルダ公開
      - `web_front\assets\i18n\<lang>\translation.json`を作成し、`web_front\assets\gui_settings\GUI.json`の`lang`に追加すれば言語対応を拡張できます。
- v.2.0.36-alpha
  - バグフィックス
    - RVCの音が割れる問題の対策
    - vcclient v1で作成したDDPN版RVCのonnxの読み込み失敗の対策
- v.2.0.32-alpha Colab版 リリース。⇒[こちら](./w_okada's_Voice_Changer_version_2_x.ipynb)
  - ngrokフリーになりました。ngrokのアカウントなしで利用可能です。
- v.2.0.27-alpha
  - Feature
    - Beatrice v2 alpha2対応：formant変更、品質向上
  - ログ強化
    - ダウンロードボタン追加
  - 改善
    - アップロードの２度押し回避
    - アップロード中の表示
    - paththrough -> passthrough
  - バグフィックス
    - performance monitorにundefinedが返ってきたときの対応追加

- v.2.0.24-alpha Colab版 リリース。⇒[こちら](./w_okada's_Voice_Changer_version_2_x.ipynb)
- v.2.0.24-alpha
  - bugfix:
    - モード切替をしたときに音が出なくなる問題を対策
  - その他：
    - loggerの強化
    - エラー画面の強化

- v.2.0.23-alpha
  - エディションを再整理
    - win_std:  一般的なwinユーザ向け。onnxモデル, torchモデルともに、DirectMLによりGPUのハードウェアアクセラレーションが可能です。
    - win_cuda:NvidiaのGPU所有者向け。onnxモデル, torchモデルともに、cudaによりNvidiaのGPUのハードウェアアクセラレーションが可能です。要cuda12.4~。
    - mac: AppleSilicon(M1等)ユーザ向け。            
  - feature
    - クライアントモードでの動作時のアウトプットバッファの調整機能を追加
  - bugfix:
    - RVCのtorchモデルをonnxモデルにエクスポートする際にindex, iconを引き継ぐように修正
  - その他：
    - loggerの強化

- v.2.0.20-alpha
  - torch-cudaに対応。エディションの説明は[こちら](docs/01_basic_v2.0.z.md)。
  - bugfix:
    - ファイルエンコーディングをUTF-8に統一

- v.2.0.16-alpha
  - torch-dmlに対応（実験的なバージョン）。エディションの説明は[こちら](docs/01_basic_v2.0.z.md)。
  - bugfix:
    - rvc file uploadの際、pthとindexの両方をアップできない不具合の対策。

- v.2.0.13-alpha
  - onnxruntime-gpuに対応。cudaエディションのリリース。エディションの説明は[こちら](docs/01_basic_v2.0.z.md)。
  - bugfix:
    - onnxcrepeの不具合対策
    - Beatrice v2 APIのID選択不具合対策
  - その他：
    - loggerの強化


- v.2.0.6-alpha
  - 新規
    - M1系 Macに対応しました。
      - M1 MBA(monterey), M2 Pro MBP(venture)での動作実績あります。
      - sonomaでのレポートお待ちしております。
  - bugfix:
    - beatriceのスピーカー選択でpitchが元に戻ってしまうバグに対応。
  - その他：
    - 不具合解析用の情報取得強化

- v.2.0.5-alpha
  - VCClientがセカンドバージョンとしてリブートしました。
  - 大幅なソフトウェア構造変更により拡張容易性を高めました。
  - REST APIを提供することでサードパーティによるクライアント開発を容易化しました。
  - エディション体系を刷新しました。
    - スタンダードエディション(win)はgpuの有無にかかわらず、onnxモデルでの実行を基本としてます。torchモデルはonnxモデルに変換してから使用してください。gpuをお持ちの方はonnxモデルでのみハードウェアアクセラレーションが有効となります。
    - cudaエディション(win)は、NvidiaのGPUに特化したチューニングがされています。スタンダードエディションと比較してさらなる高速化が可能です。onnxモデルでのみハードウェアアクセラレーションが有効となります。
    - torchモデルはpytorchのモデルもハードウェアアクセラレートできます。
    - macエディションはApple Silicon搭載のMacユーザ向けです。
    - linuxユーザやpythonの知識がある方はリポジトリをcloneして実行することもできます。
  - 現在のalpha versionではスタンダードエディションのみの提供となっています。
  

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
