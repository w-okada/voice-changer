[Bahasa Jepun](/README.md) /
[Bahasa Inggeris](/docs_i18n/README_en.md) /
[Bahasa Korea](/docs_i18n/README_ko.md)/
[Bahasa Cina](/docs_i18n/README_zh.md)/
[Bahasa Jerman](/docs_i18n/README_de.md)/
[Bahasa Arab](/docs_i18n/README_ar.md)/
[Bahasa Greek](/docs_i18n/README_el.md)/
[Bahasa Sepanyol](/docs_i18n/README_es.md)/
[Bahasa Perancis](/docs_i18n/README_fr.md)/
[Bahasa Itali](/docs_i18n/README_it.md)/
[Bahasa Latin](/docs_i18n/README_la.md)/
[Bahasa Melayu](/docs_i18n/README_ms.md)/
[Bahasa Rusia](/docs_i18n/README_ru.md)
*Selain bahasa Jepun, semua terjemahan adalah terjemahan mesin.

## VCClient

VCClient adalah perisian yang menggunakan AI untuk menukar suara secara masa nyata.

## Apa yang Baru!

* v.2.0.76-beta
  * ciri baru:
    * Beatrice: Pelaksanaan penggabungan pembicara
    * Beatrice: Auto pitch shift
  * pembaikan pepijat:
    * Menangani masalah pemilihan peranti dalam mod pelayan
* v.2.0.73-beta
  * ciri baru:
    * Muat turun model beatrice yang telah diedit
  * pembaikan pepijat:
    * Memperbaiki pepijat di mana pitch dan formant beatrice v2 tidak diterapkan
    * Memperbaiki pepijat di mana ONNX tidak dapat dibuat untuk model yang menggunakan embedder Applio

## Muat Turun dan Pautan Berkaitan

Versi Windows dan M1 Mac boleh dimuat turun dari repositori hugging face.

* [Repositori VCClient](https://huggingface.co/wok000/vcclient000/tree/main)
* [Repositori Light VCClient untuk Beatrice v2](https://huggingface.co/wok000/light_vcclient_beatrice/tree/main)

*1 Sila klon repositori untuk Linux.

### Pautan Berkaitan

* [Repositori Kod Latihan Beatrice V2](https://huggingface.co/fierce-cats/beatrice-trainer)
* [Versi Colab Kod Latihan Beatrice V2](https://github.com/w-okada/beatrice-trainer-colab)

### Perisian Berkaitan

* [Penukar Suara Masa Nyata VCClient](https://github.com/w-okada/voice-changer)
* [Perisian Pembacaan TTSClient](https://github.com/w-okada/ttsclient)
* [Perisian Pengecaman Suara Masa Nyata ASRClient](https://github.com/w-okada/asrclient)

## Ciri-ciri VC Client

## Menyokong pelbagai model AI

| Model AI                                                                                                     | v.2       | v.1                  | Lesen                                                                                 |
| ------------------------------------------------------------------------------------------------------------ | --------- | -------------------- | ------------------------------------------------------------------------------------------ |
| [RVC ](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/jp/README.ja.md) | disokong | disokong            | Sila rujuk repositori.                                                             |
| [Beatrice v1](https://prj-beatrice.com/)                                                                     | n/a       | disokong (hanya win) | [Khas](https://github.com/w-okada/voice-changer/tree/master/server/voice_changer/Beatrice) |
| [Beatrice v2](https://prj-beatrice.com/)                                                                     | disokong | n/a                  | [Khas](https://huggingface.co/wok000/vcclient_model/blob/main/beatrice_v2_beta/readme.md)  |
| [MMVC](https://github.com/isletennos/MMVC_Trainer)                                                           | n/a       | disokong            | Sila rujuk repositori.                                                             |
| [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)                                               | n/a       | disokong            | Sila rujuk repositori.                                                             |
| [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)                                                               | n/a       | disokong            | Sila rujuk repositori.                                                             |

## Menyokong kedua-dua konfigurasi berdiri sendiri dan melalui rangkaian

Menyokong penukaran suara yang lengkap di PC tempatan dan juga melalui rangkaian.
Dengan menggunakan melalui rangkaian, beban penukaran suara boleh dialihkan ke luar apabila digunakan serentak dengan aplikasi yang memerlukan beban tinggi seperti permainan.

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

## Menyokong pelbagai platform

Windows, Mac(M1), Linux, Google Colab

*1 Sila klon repositori untuk Linux.

## Menyediakan REST API

Pelanggan boleh dibina dalam pelbagai bahasa pengaturcaraan.

Juga boleh dikendalikan menggunakan klien HTTP yang dibina dalam OS seperti curl.

## Penyelesaian Masalah

[Bahagian Komunikasi](tutorials/trouble_shoot_communication_ja.md)

## Mengenai Tandatangan Pembangun

Perisian ini tidak ditandatangani oleh pembangun. Amaran seperti di bawah akan muncul, tetapi anda boleh menjalankannya dengan menekan kekunci kawalan sambil mengklik ikon. Ini adalah disebabkan oleh dasar keselamatan Apple. Pelaksanaan adalah atas tanggungjawab sendiri.

![image](https://user-images.githubusercontent.com/48346627/212567711-c4a8d599-e24c-4fa3-8145-a5df7211f023.png)

## Penghargaan

* [Bahan Tachizundamon](https://seiga.nicovideo.jp/seiga/im10792934)
* [Irasutoya](https://www.irasutoya.com/)
* [Tsukuyomi-chan](https://tyc.rei-yumesaki.net/)

```
  本ソフトウェアの音声合成には、フリー素材キャラクター「つくよみちゃん」が無料公開している音声データを使用しています。
  ■つくよみちゃんコーパス（CV.夢前黎）
  https://tyc.rei-yumesaki.net/material/corpus/
  © Rei Yumesaki
```

* [Studio Bahan Suara Amitaro](https://amitaro.net/)
* [Replikadol](https://kikyohiroto1227.wixsite.com/kikoto-utau)

## Syarat Penggunaan

* Mengenai penukar suara masa nyata Tsukuyomi-chan, penggunaan suara yang ditukar untuk tujuan berikut adalah dilarang mengikut syarat penggunaan korpus Tsukuyomi-chan.

```

■人を批判・攻撃すること。（「批判・攻撃」の定義は、つくよみちゃんキャラクターライセンスに準じます）

■特定の政治的立場・宗教・思想への賛同または反対を呼びかけること。

■刺激の強い表現をゾーニングなしで公開すること。

■他者に対して二次利用（素材としての利用）を許可する形で公開すること。
※鑑賞用の作品として配布・販売していただくことは問題ございません。
```

* Mengenai penukar suara masa nyata Amitaro, ia mematuhi syarat penggunaan Studio Bahan Suara Amitaro. Untuk maklumat lanjut, sila lihat[di sini](https://amitaro.net/voice/faq/#index_id6)

```
あみたろの声素材やコーパス読み上げ音声を使って音声モデルを作ったり、ボイスチェンジャーや声質変換などを使用して、自分の声をあみたろの声に変換して使うのもOKです。

ただしその場合は絶対に、あみたろ（もしくは小春音アミ）の声に声質変換していることを明記し、あみたろ（および小春音アミ）が話しているわけではないことが誰でもわかるようにしてください。
また、あみたろの声で話す内容は声素材の利用規約の範囲内のみとし、センシティブな発言などはしないでください。
```

* Mengenai penukar suara masa nyata Kogane Mahiro, ia mematuhi syarat penggunaan Replikadol. Untuk maklumat lanjut, sila lihat[di sini](https://kikyohiroto1227.wixsite.com/kikoto-utau/ter%EF%BD%8Ds-of-service)

## Penafian

Kami tidak bertanggungjawab ke atas sebarang kerosakan langsung, tidak langsung, berbangkit, akibat atau khas yang timbul daripada penggunaan atau ketidakupayaan untuk menggunakan perisian ini.
