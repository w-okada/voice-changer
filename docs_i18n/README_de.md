[Japanisch](/README.md) /
[Englisch](/docs_i18n/README_en.md) /
[Koreanisch](/docs_i18n/README_ko.md)/
[Chinesisch](/docs_i18n/README_zh.md)/
[Deutsch](/docs_i18n/README_de.md)/
[Arabisch](/docs_i18n/README_ar.md)/
[Griechisch](/docs_i18n/README_el.md)/
[Spanisch](/docs_i18n/README_es.md)/
[Französisch](/docs_i18n/README_fr.md)/
[Italienisch](/docs_i18n/README_it.md)/
[Latein](/docs_i18n/README_la.md)/
[Malaiisch](/docs_i18n/README_ms.md)/
[Russisch](/docs_i18n/README_ru.md)
*Außer Japanisch sind alle Übersetzungen maschinell.

## VCClient

VCClient ist eine Software, die mithilfe von KI eine Echtzeit-Sprachumwandlung durchführt.

## What's New!

* v.2.2.2-beta
  * Veröffentlichte Editionen: std_win, std_mac, std_lin_aarch64
  * Ab v.2.2.2-beta unterscheiden sich die unterstützten Modelle je nach Edition (siehe unten). Zudem können sich die veröffentlichten Editionen je nach Version unterscheiden.
  * Unterstützt Beatrice v2.0.0-rc0.
* v.2.0.78-beta
  * Fehlerbehebung: Upload-Fehler für RVC-Modell vermieden
  * Gleichzeitiger Start mit Version 1.x jetzt möglich
  * Auswahlbare Chunk-Größen erhöht
* v.2.0.77-beta (nur für RTX 5090, experimentell)
  * Unterstützung für RTX 5090 verwandte Module (nicht verifiziert, da Entwickler kein RTX 5090 besitzt)
* v.2.0.76-beta
  * neues Feature:
    * Beatrice: Implementierung der Sprecherzusammenführung
    * Beatrice: Automatische Tonhöhenverschiebung
  * Fehlerbehebung:
    * Problembehebung bei der Gerätauswahl im Servermodus
* v.2.0.73-beta
  * neues Feature:
    * Download des bearbeiteten Beatrice-Modells
  * Fehlerbehebung:
    * Fehler behoben, bei dem Pitch und Formant von Beatrice v2 nicht reflektiert wurden
    * Fehler behoben, bei dem das ONNX-Modell mit dem Applio-Embedder nicht erstellt werden konnte

## Editionen

Ab v2.2.1 unterstützen die Editionen unterschiedliche KI-Modelle.

| Edition | Plattform      | Unterstützte Modelle |     |
| ------- | -------------- | -------------------- | --- |
| std     | win            | Beatrice             |     |
| std     | mac(m1)        | Beatrice             |     |
| std     | linux(x86-64)  | Beatrice             |     |
| std     | linux(aarch64) | Beatrice             |     |
| cuda    | win            | Beatrice, RVC        |     |
| onnx    | win            | Beatrice, RVC        |     |
| onnx    | mac(m1)        | Beatrice, RVC        |     |

## Downloads und verwandte Links

Windows- und M1 Mac-Versionen können aus dem Repository von Hugging Face heruntergeladen werden.

* [VCClient-Repository](https://huggingface.co/wok000/vcclient000/tree/main)
* [Light VCClient für Beatrice v2 Repository](https://huggingface.co/wok000/light_vcclient_beatrice/tree/main)

*1 Linux: Bitte klonen Sie das Repository zur Nutzung.

### Verwandte Links

* [Beatrice V2 Trainingscode-Repository](https://huggingface.co/fierce-cats/beatrice-trainer)
* [Beatrice V2 Trainingscode Colab-Version](https://github.com/w-okada/beatrice-trainer-colab)

### Verwandte Software

* [Echtzeit-Voice-Changer VCClient](https://github.com/w-okada/voice-changer)
* [Vorlesesoftware TTSClient](https://github.com/w-okada/ttsclient)
* [Echtzeit-Spracherkennungssoftware ASRClient](https://github.com/w-okada/asrclient)

## Merkmale des VC Clients

## Unterstützt verschiedene KI-Modelle

| KI-Modelle                                                                                                     | v.2       | v.1                  | Lizenz                                                                                 |
| ------------------------------------------------------------------------------------------------------------ | --------- | -------------------- | ------------------------------------------------------------------------------------------ |
| [RVC ](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/jp/README.ja.md) | unterstützt | unterstützt            | Bitte das Repository konsultieren.                                                             |
| [Beatrice v1](https://prj-beatrice.com/)                                                                     | n/a       | unterstützt (nur Windows) | [Eigen](https://github.com/w-okada/voice-changer/tree/master/server/voice_changer/Beatrice) |
| [Beatrice v2](https://prj-beatrice.com/)                                                                     | unterstützt | n/a                  | [Eigen](https://huggingface.co/wok000/vcclient_model/blob/main/beatrice_v2_beta/readme.md)  |
| [MMVC](https://github.com/isletennos/MMVC_Trainer)                                                           | n/a       | unterstützt            | Bitte das Repository konsultieren.                                                             |
| [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)                                               | n/a       | unterstützt            | Bitte das Repository konsultieren.                                                             |
| [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)                                                               | n/a       | unterstützt            | Bitte das Repository konsultieren.                                                             |

## Unterstützt sowohl Standalone- als auch Netzwerk-Konfigurationen

Unterstützt sowohl Sprachumwandlung auf dem lokalen PC als auch über das Netzwerk.
Durch die Nutzung über das Netzwerk kann die Belastung der Sprachumwandlung auf externe Ressourcen ausgelagert werden, wenn gleichzeitig ressourcenintensive Anwendungen wie Spiele genutzt werden.

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

## Unterstützt mehrere Plattformen

Windows, Mac(M1), Linux, Google Colab

*1 Linux: Bitte klonen Sie das Repository zur Nutzung.

## Bietet REST API

Clients können in verschiedenen Programmiersprachen erstellt werden.

Außerdem kann die Bedienung mit in das Betriebssystem integrierten HTTP-Clients wie curl erfolgen.

## Fehlerbehebung

[Kommunikationsprobleme](tutorials/trouble_shoot_communication_ja.md)

## Über die Signatur des Entwicklers

Diese Software ist nicht vom Entwickler signiert. Es wird eine Warnung wie unten angezeigt, aber Sie können sie ausführen, indem Sie die Steuerungstaste gedrückt halten und auf das Symbol klicken. Dies liegt an den Sicherheitsrichtlinien von Apple. Die Ausführung erfolgt auf eigenes Risiko.

![image](https://user-images.githubusercontent.com/48346627/212567711-c4a8d599-e24c-4fa3-8145-a5df7211f023.png)

## Danksagungen

* [Tachizundamon-Material](https://seiga.nicovideo.jp/seiga/im10792934)
* [Irasutoya](https://www.irasutoya.com/)
* [Tsukuyomi-chan](https://tyc.rei-yumesaki.net/)

```
  本ソフトウェアの音声合成には、フリー素材キャラクター「つくよみちゃん」が無料公開している音声データを使用しています。
  ■つくよみちゃんコーパス（CV.夢前黎）
  https://tyc.rei-yumesaki.net/material/corpus/
  © Rei Yumesaki
```

* [Amitaro's Voice Material Studio](https://amitaro.net/)
* [Replikador](https://kikyohiroto1227.wixsite.com/kikoto-utau)

## Nutzungsbedingungen

* Für den Echtzeit-Voice-Changer Tsukuyomi-chan gelten die Nutzungsbedingungen des Tsukuyomi-chan-Korpus, und die Verwendung der umgewandelten Stimme für die folgenden Zwecke ist untersagt.

```

■人を批判・攻撃すること。（「批判・攻撃」の定義は、つくよみちゃんキャラクターライセンスに準じます）

■特定の政治的立場・宗教・思想への賛同または反対を呼びかけること。

■刺激の強い表現をゾーニングなしで公開すること。

■他者に対して二次利用（素材としての利用）を許可する形で公開すること。
※鑑賞用の作品として配布・販売していただくことは問題ございません。
```

* Für den Echtzeit-Voice-Changer Amitaro gelten die folgenden Nutzungsbedingungen von Amitaro's Voice Material Studio. Details finden Sie[hier](https://amitaro.net/voice/faq/#index_id6)

```
あみたろの声素材やコーパス読み上げ音声を使って音声モデルを作ったり、ボイスチェンジャーや声質変換などを使用して、自分の声をあみたろの声に変換して使うのもOKです。

ただしその場合は絶対に、あみたろ（もしくは小春音アミ）の声に声質変換していることを明記し、あみたろ（および小春音アミ）が話しているわけではないことが誰でもわかるようにしてください。
また、あみたろの声で話す内容は声素材の利用規約の範囲内のみとし、センシティブな発言などはしないでください。
```

* Für den Echtzeit-Voice-Changer Koto Mahiro gelten die Nutzungsbedingungen von Replikador. Details finden Sie[hier](https://kikyohiroto1227.wixsite.com/kikoto-utau/ter%EF%BD%8Ds-of-service)

## Haftungsausschluss

Wir übernehmen keine Verantwortung für direkte, indirekte, Folgeschäden, resultierende oder besondere Schäden, die durch die Nutzung oder Unfähigkeit zur Nutzung dieser Software entstehen.

```
