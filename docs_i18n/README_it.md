[Giapponese](/README.md) /
[Inglese](/docs_i18n/README_en.md) /
[Coreano](/docs_i18n/README_ko.md)/
[Cinese](/docs_i18n/README_zh.md)/
[Tedesco](/docs_i18n/README_de.md)/
[Arabo](/docs_i18n/README_ar.md)/
[Greco](/docs_i18n/README_el.md)/
[Spagnolo](/docs_i18n/README_es.md)/
[Francese](/docs_i18n/README_fr.md)/
[Italiano](/docs_i18n/README_it.md)/
[Latino](/docs_i18n/README_la.md)/
[Malese](/docs_i18n/README_ms.md)/
[Russo](/docs_i18n/README_ru.md)
*Le lingue diverse dal giapponese sono tradotte automaticamente.

## VCClient

VCClient è un software che utilizza l'IA per la conversione vocale in tempo reale.

## Novità!

* v.2.0.73-beta
  * nuova funzionalità:
    * Download del modello beatrice modificato
  * correzione bug:
    * Corretto un bug per cui pitch e formant di beatrice v2 non venivano applicati
    * Corretto un bug per cui non era possibile creare ONNX per i modelli che utilizzano l'embedder di Applio
* v.2.0.72-beta (sconsigliato a causa di bug, corretto in v.2.0.73)
  * nuova funzionalità
    * Interfaccia GUI di modifica di Beatrice
    * Memorizzazione di pitch e formant per ogni voce di Beatrice
    * Interfaccia GUI multilingue
    * Supporto per embedder di Applio
* v.2.0.70-beta (solo per Mac M1)
  * nuova funzionalità:
    * Anche la versione VCClient per Mac M1 supporta Beatrice v2 beta.1.
* v.2.0.69-beta (solo per Windows)
  * correzione bug:
    * Corretto il bug per cui il pulsante di avvio non veniva visualizzato in caso di alcune eccezioni
    * Regolato il buffer di output in modalità dispositivo server
    * Corretto il bug per cui la frequenza di campionamento cambiava quando si modificavano le impostazioni durante l'uso della modalità dispositivo server
    * Correzione bug per l'uso di hubert giapponese
  * varie:
    * Aggiunto filtro API host in modalità dispositivo server (evidenziato)

## Download e link correlati

Le versioni per Windows e Mac M1 possono essere scaricate dal repository di hugging face.

* [Repository di VCClient](https://huggingface.co/wok000/vcclient000/tree/main)
* [Repository di Light VCClient per Beatrice v2](https://huggingface.co/wok000/light_vcclient_beatrice/tree/main)

*1 Per Linux, clona il repository per l'uso.

### Link correlati

* [Repository del codice di allenamento Beatrice V2](https://huggingface.co/fierce-cats/beatrice-trainer)
* [Versione Colab del codice di allenamento Beatrice V2](https://github.com/w-okada/beatrice-trainer-colab)

### Software correlato

* [Cambiavoce in tempo reale VCClient](https://github.com/w-okada/voice-changer)
* [Software di sintesi vocale TTSClient](https://github.com/w-okada/ttsclient)
* [Software di riconoscimento vocale in tempo reale ASRClient](https://github.com/w-okada/asrclient)

## Caratteristiche di VC Client

## Supporta vari modelli di IA

| Modello di IA                                                                                                     | v.2       | v.1                  | Licenza                                                                                 |
| ------------------------------------------------------------------------------------------------------------ | --------- | -------------------- | ------------------------------------------------------------------------------------------ |
| [RVC ](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/jp/README.ja.md) | supportato | supportato            | Si prega di consultare il repository.                                                             |
| [Beatrice v1](https://prj-beatrice.com/)                                                                     | n/a       | supportato (solo win) | [Proprietario](https://github.com/w-okada/voice-changer/tree/master/server/voice_changer/Beatrice) |
| [Beatrice v2](https://prj-beatrice.com/)                                                                     | supportato | n/a                  | [Proprietario](https://huggingface.co/wok000/vcclient_model/blob/main/beatrice_v2_beta/readme.md)  |
| [MMVC](https://github.com/isletennos/MMVC_Trainer)                                                           | n/a       | supportato            | Si prega di consultare il repository.                                                             |
| [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)                                               | n/a       | supportato            | Si prega di consultare il repository.                                                             |
| [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)                                                               | n/a       | supportato            | Si prega di consultare il repository.                                                             |

## Supporta sia la configurazione standalone che tramite rete

Supporta sia la conversione vocale completata su PC locale che tramite rete.
Utilizzando tramite rete, è possibile scaricare il carico della conversione vocale su un dispositivo esterno quando si utilizzano applicazioni ad alto carico come i giochi.

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

## Compatibile con più piattaforme

Windows, Mac(M1), Linux, Google Colab

*1 Per Linux, clona il repository per l'uso.

## Fornisce un'API REST

È possibile creare client in vari linguaggi di programmazione.

È inoltre possibile operare utilizzando client HTTP incorporati nel sistema operativo come curl.

## Risoluzione dei problemi

[Sezione comunicazione](tutorials/trouble_shoot_communication_ja.md)

## Informazioni sulla firma dello sviluppatore

Questo software non è firmato dallo sviluppatore. Anche se viene visualizzato un avviso come di seguito, è possibile eseguirlo facendo clic sull'icona tenendo premuto il tasto di controllo. Questo è dovuto alla politica di sicurezza di Apple. L'esecuzione è a proprio rischio.

![image](https://user-images.githubusercontent.com/48346627/212567711-c4a8d599-e24c-4fa3-8145-a5df7211f023.png)

## Ringraziamenti

* [Materiale di Tachi Zundamon](https://seiga.nicovideo.jp/seiga/im10792934)
* [Irasutoya](https://www.irasutoya.com/)
* [Tsukuyomi-chan](https://tyc.rei-yumesaki.net/)

```
  本ソフトウェアの音声合成には、フリー素材キャラクター「つくよみちゃん」が無料公開している音声データを使用しています。
  ■つくよみちゃんコーパス（CV.夢前黎）
  https://tyc.rei-yumesaki.net/material/corpus/
  © Rei Yumesaki
```

* [Atelier di materiali vocali di Amitaro](https://amitaro.net/)
* [Replica Doll](https://kikyohiroto1227.wixsite.com/kikoto-utau)

## Termini di utilizzo

* Per quanto riguarda il cambiavoce in tempo reale Tsukuyomi-chan, è vietato utilizzare la voce convertita per i seguenti scopi in conformità con i termini di utilizzo del corpus di Tsukuyomi-chan.

```

■人を批判・攻撃すること。（「批判・攻撃」の定義は、つくよみちゃんキャラクターライセンスに準じます）

■特定の政治的立場・宗教・思想への賛同または反対を呼びかけること。

■刺激の強い表現をゾーニングなしで公開すること。

■他者に対して二次利用（素材としての利用）を許可する形で公開すること。
※鑑賞用の作品として配布・販売していただくことは問題ございません。
```

* Per quanto riguarda il cambiavoce in tempo reale Amitaro, si applicano i seguenti termini di utilizzo dell'Atelier di materiali vocali di Amitaro. Per dettagli, [qui](https://amitaro.net/voice/faq/#index_id6)

```
あみたろの声素材やコーパス読み上げ音声を使って音声モデルを作ったり、ボイスチェンジャーや声質変換などを使用して、自分の声をあみたろの声に変換して使うのもOKです。

ただしその場合は絶対に、あみたろ（もしくは小春音アミ）の声に声質変換していることを明記し、あみたろ（および小春音アミ）が話しているわけではないことが誰でもわかるようにしてください。
また、あみたろの声で話す内容は声素材の利用規約の範囲内のみとし、センシティブな発言などはしないでください。
```

* Per quanto riguarda il cambiavoce in tempo reale Koto Mahiro, si applicano i termini di utilizzo di Replica Doll. Per dettagli, [qui](https://kikyohiroto1227.wixsite.com/kikoto-utau/ter%EF%BD%8Ds-of-service)

## Clausola di esclusione della responsabilità

Non ci assumiamo alcuna responsabilità per eventuali danni diretti, indiretti, consequenziali, risultanti o speciali derivanti dall'uso o dall'impossibilità di utilizzare questo software.
