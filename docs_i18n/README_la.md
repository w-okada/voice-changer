[Lingua Iaponica](/README.md) /
[Lingua Anglica](/docs_i18n/README_en.md) /
[Lingua Coreana](/docs_i18n/README_ko.md)/
[Lingua Sinica](/docs_i18n/README_zh.md)/
[Lingua Theodisca](/docs_i18n/README_de.md)/
[Lingua Arabica](/docs_i18n/README_ar.md)/
[Lingua Graeca](/docs_i18n/README_el.md)/
[Lingua Hispanica](/docs_i18n/README_es.md)/
[Lingua Francogallica](/docs_i18n/README_fr.md)/
[Lingua Italica](/docs_i18n/README_it.md)/
[Lingua Latina](/docs_i18n/README_la.md)/
[Lingua Malaica](/docs_i18n/README_ms.md)/
[Lingua Russica](/docs_i18n/README_ru.md)
*Praeter linguam Iaponicam, omnes linguae sunt a machina translatae.

## VCClient

VCClient est software quod conversionem vocis in tempore reali per AI facit.

## Quid Novum!

* v.2.0.72-beta
  * nova functio
    * GUI editionis Beatrice
    * Memoria pitch et formant pro singulis vocibus Beatrice
    * GUI multilanguage
    * Embedder Applio compatibilitas
* v.2.0.70-beta (solum pro m1 mac)
  * nova functio:
    * In versione M1 Mac VCClient etiam Beatrice v2 beta.1 sustinetur.
* v.2.0.69-beta (solum pro win)
  * bugfix:
    * Errorem correximus ubi puga initii non apparebat in casu exceptionis.
    * Buffer outputis in modo machinae servientis adaptavimus.
    * Errorem correximus ubi mutatio configurationis in modo machinae servientis rate sampling mutabat.
    * Errorem cum usura Iaponica hubert correximus.
  * misc:
    * Filtrum API hospitis in modo machinae servientis additum (emphasis)

## Download et nexus pertinentes

Versiones pro Windows et M1 Mac possunt ex repositorio hugging face depromi.

* [Repositorium VCClient](https://huggingface.co/wok000/vcclient000/tree/main)
* [Repositorium Light VCClient pro Beatrice v2](https://huggingface.co/wok000/light_vcclient_beatrice/tree/main)

*1 Linux utatur repositorio clone.

### Nexus pertinentes

* [Repositorium codicis disciplinae Beatrice V2](https://huggingface.co/fierce-cats/beatrice-trainer)
* [Codex disciplinae Beatrice V2 versio Colab](https://github.com/w-okada/beatrice-trainer-colab)

### Software pertinens

* [Mutator vocis in tempore reali VCClient](https://github.com/w-okada/voice-changer)
* [Software lectionis TTSClient](https://github.com/w-okada/ttsclient)
* [Software recognitionis vocis in tempore reali ASRClient](https://github.com/w-okada/asrclient)

## Proprietates VC Client

## Multa AI exempla sustinet

| Exempla AI                                                                                                     | v.2       | v.1                  | Licentia                                                                                 |
| ------------------------------------------------------------------------------------------------------------ | --------- | -------------------- | ------------------------------------------------------------------------------------------ |
| [RVC ](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/jp/README.ja.md) | sustinetur | sustinetur            | Vide repositorium.                                                             |
| [Beatrice v1](https://prj-beatrice.com/)                                                                     | n/a       | sustinetur (solum win) | [Proprium](https://github.com/w-okada/voice-changer/tree/master/server/voice_changer/Beatrice) |
| [Beatrice v2](https://prj-beatrice.com/)                                                                     | sustinetur | n/a                  | [Proprium](https://huggingface.co/wok000/vcclient_model/blob/main/beatrice_v2_beta/readme.md)  |
| [MMVC](https://github.com/isletennos/MMVC_Trainer)                                                           | n/a       | sustinetur            | Vide repositorium.                                                             |
| [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)                                               | n/a       | sustinetur            | Vide repositorium.                                                             |
| [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)                                                               | n/a       | sustinetur            | Vide repositorium.                                                             |

## Sustinetur tam structura stand-alone quam per rete

Sustinetur conversio vocis in PC locali et per rete.
Per usum per rete, onus conversionis vocis potest externari cum simul cum applicationibus altis oneribus ut ludis adhibetur.

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

## Pluribus suggestis compatitur

Windows, Mac(M1), Linux, Google Colab

*1 Linux utatur repositorio clone.

## REST API praebet

Clientem creare potes in variis linguis programmandi.

Etiam per HTTP clientem in OS incorporatum ut curl operari potes.

## Solutio problematum

[De communicatione](tutorials/trouble_shoot_communication_ja.md)

## De signature auctoris

Hoc software non signatur auctore. Monitio ut infra apparebit, sed si iconem cum claviatura control premes, poteris exsequi. Hoc est secundum securitatem Apple. Exsecutio est tuae responsabilitatis.

![image](https://user-images.githubusercontent.com/48346627/212567711-c4a8d599-e24c-4fa3-8145-a5df7211f023.png)

## Gratias

* [Materia Tachi Zundamon](https://seiga.nicovideo.jp/seiga/im10792934)
* [Irasuto ya](https://www.irasutoya.com/)
* [Tsukuyomi-chan](https://tyc.rei-yumesaki.net/)

```
  本ソフトウェアの音声合成には、フリー素材キャラクター「つくよみちゃん」が無料公開している音声データを使用しています。
  ■つくよみちゃんコーパス（CV.夢前黎）
  https://tyc.rei-yumesaki.net/material/corpus/
  © Rei Yumesaki
```

* [Amitaro vox materiae officina](https://amitaro.net/)
* [Reprica doll](https://kikyohiroto1227.wixsite.com/kikoto-utau)

## Termini usus

* De mutatore vocis in tempore reali Tsukuyomi-chan, secundum Tsukuyomi-chan corpus usus, prohibetur usus vocis post conversionem ad sequentes fines.

```

■人を批判・攻撃すること。（「批判・攻撃」の定義は、つくよみちゃんキャラクターライセンスに準じます）

■特定の政治的立場・宗教・思想への賛同または反対を呼びかけること。

■刺激の強い表現をゾーニングなしで公開すること。

■他者に対して二次利用（素材としての利用）を許可する形で公開すること。
※鑑賞用の作品として配布・販売していただくことは問題ございません。
```

* De mutatore vocis in tempore reali Amitaro, secundum Amitaro vox materiae officinae usus. Pro details[hic](https://amitaro.net/voice/faq/#index_id6)

```
あみたろの声素材やコーパス読み上げ音声を使って音声モデルを作ったり、ボイスチェンジャーや声質変換などを使用して、自分の声をあみたろの声に変換して使うのもOKです。

ただしその場合は絶対に、あみたろ（もしくは小春音アミ）の声に声質変換していることを明記し、あみたろ（および小春音アミ）が話しているわけではないことが誰でもわかるようにしてください。
また、あみたろの声で話す内容は声素材の利用規約の範囲内のみとし、センシティブな発言などはしないでください。
```

* De mutatore vocis in tempore reali Kogane Mahiro, secundum Reprica doll usus. Pro details[hic](https://kikyohiroto1227.wixsite.com/kikoto-utau/ter%EF%BD%8Ds-of-service)

## Disclaimer

Non tenemur pro ullis damnis directis, indirectis, consequentibus, vel specialibus ex usu vel incapacitate usus huius software.
