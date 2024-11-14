[Japonés](/README.md) /
[Inglés](/docs_i18n/README_en.md) /
[Coreano](/docs_i18n/README_ko.md)/
[Chino](/docs_i18n/README_zh.md)/
[Alemán](/docs_i18n/README_de.md)/
[Árabe](/docs_i18n/README_ar.md)/
[Griego](/docs_i18n/README_el.md)/
[Español](/docs_i18n/README_es.md)/
[Francés](/docs_i18n/README_fr.md)/
[Italiano](/docs_i18n/README_it.md)/
[Latín](/docs_i18n/README_la.md)/
[Malayo](/docs_i18n/README_ms.md)/
[Ruso](/docs_i18n/README_ru.md)
*Los idiomas distintos al japonés son traducciones automáticas.

## VCClient

VCClient es un software que utiliza IA para realizar conversión de voz en tiempo real.

## ¡Novedades!

* v.2.0.73-beta
  * nueva característica:
    * Descarga del modelo Beatrice editado
  * corrección de errores:
    * Se corrigió un error donde el pitch y el formante de Beatrice v2 no se reflejaban
    * Se corrigió un error donde no se podía crear ONNX para modelos que usan el embedder de Applio
* v.2.0.72-beta (no recomendado debido a errores. Corregido en v.2.0.73)
  * nueva característica
    * GUI de edición de Beatrice
    * Memoriza pitch y formante por cada voz de Beatrice
    * Interfaz multilingüe
    * Compatibilidad con embedder de Applio
* v.2.0.70-beta (solo para Mac m1)
  * nueva característica:
    * Ahora se admite Beatrice v2 beta.1 en la versión de VCClient para Mac M1.
* v.2.0.69-beta (solo para win)
  * corrección de errores:
    * Se corrigió un error que impedía que se mostrara el botón de inicio cuando ocurrían ciertas excepciones.
    * Se ajustó el búfer de salida en modo de dispositivo de servidor.
    * Se corrigió un error que cambiaba la tasa de muestreo al modificar la configuración mientras se usaba el modo de dispositivo de servidor.
    * Corrección de errores al usar hubert en japonés
  * varios:
    * Se añadió un filtro de API de host en modo de dispositivo de servidor (resaltado)

## Descargas y enlaces relacionados

Las versiones para Windows y Mac M1 se pueden descargar desde el repositorio de hugging face.

* [Repositorio de VCClient](https://huggingface.co/wok000/vcclient000/tree/main)
* [Repositorio de Light VCClient para Beatrice v2](https://huggingface.co/wok000/light_vcclient_beatrice/tree/main)

*1 Para Linux, clone el repositorio para su uso.

### Enlaces relacionados

* [Repositorio de código de entrenamiento de Beatrice V2](https://huggingface.co/fierce-cats/beatrice-trainer)
* [Versión Colab del código de entrenamiento de Beatrice V2](https://github.com/w-okada/beatrice-trainer-colab)

### Software relacionado

* [Cambiador de voz en tiempo real VCClient](https://github.com/w-okada/voice-changer)
* [Software de lectura TTSClient](https://github.com/w-okada/ttsclient)
* [Software de reconocimiento de voz en tiempo real ASRClient](https://github.com/w-okada/asrclient)

## Características de VC Client

## Soporta diversos modelos de IA

| Modelos de IA                                                                                                     | v.2       | v.1                  | Licencia                                                                                 |
| ------------------------------------------------------------------------------------------------------------ | --------- | -------------------- | ------------------------------------------------------------------------------------------ |
| [RVC ](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/jp/README.ja.md) | soportado | soportado            | Consulte el repositorio.                                                             |
| [Beatrice v1](https://prj-beatrice.com/)                                                                     | n/a       | soportado (solo win) | [Propio](https://github.com/w-okada/voice-changer/tree/master/server/voice_changer/Beatrice) |
| [Beatrice v2](https://prj-beatrice.com/)                                                                     | soportado | n/a                  | [Propio](https://huggingface.co/wok000/vcclient_model/blob/main/beatrice_v2_beta/readme.md)  |
| [MMVC](https://github.com/isletennos/MMVC_Trainer)                                                           | n/a       | soportado            | Consulte el repositorio.                                                             |
| [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)                                               | n/a       | soportado            | Consulte el repositorio.                                                             |
| [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)                                                               | n/a       | soportado            | Consulte el repositorio.                                                             |

## Soporta configuraciones tanto autónomas como a través de la red

Soporta tanto la conversión de voz completada en una PC local como la conversión de voz a través de la red.
Al utilizarlo a través de la red, puede descargar la carga de conversión de voz externamente cuando se usa simultáneamente con aplicaciones de alta carga como juegos.

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

## Compatible con múltiples plataformas

Windows, Mac(M1), Linux, Google Colab

*1 Para Linux, clone el repositorio para su uso.

## Proporciona API REST

Puede crear clientes en varios lenguajes de programación.

Además, puede operar usando clientes HTTP integrados en el sistema operativo como curl.

## Solución de problemas

[Sección de comunicación](tutorials/trouble_shoot_communication_ja.md)

## Sobre la firma del desarrollador

Este software no está firmado por el desarrollador. Aunque aparece una advertencia como se muestra a continuación, puede ejecutarlo haciendo clic en el icono mientras mantiene presionada la tecla de control. Esto se debe a la política de seguridad de Apple. La ejecución es bajo su propio riesgo.

![image](https://user-images.githubusercontent.com/48346627/212567711-c4a8d599-e24c-4fa3-8145-a5df7211f023.png)

## Agradecimientos

* [Material de Tachi Zundamon](https://seiga.nicovideo.jp/seiga/im10792934)
* [Ilustraciones de Irasutoya](https://www.irasutoya.com/)
* [Tsukuyomi-chan](https://tyc.rei-yumesaki.net/)

```
  本ソフトウェアの音声合成には、フリー素材キャラクター「つくよみちゃん」が無料公開している音声データを使用しています。
  ■つくよみちゃんコーパス（CV.夢前黎）
  https://tyc.rei-yumesaki.net/material/corpus/
  © Rei Yumesaki
```

* [Taller de voz de Amitaro](https://amitaro.net/)
* [Replikador](https://kikyohiroto1227.wixsite.com/kikoto-utau)

## Términos de uso

* En cuanto a Tsukuyomi-chan, el cambiador de voz en tiempo real, está prohibido usar la voz convertida para los siguientes propósitos, de acuerdo con los términos de uso del corpus de Tsukuyomi-chan.

```

■人を批判・攻撃すること。（「批判・攻撃」の定義は、つくよみちゃんキャラクターライセンスに準じます）

■特定の政治的立場・宗教・思想への賛同または反対を呼びかけること。

■刺激の強い表現をゾーニングなしで公開すること。

■他者に対して二次利用（素材としての利用）を許可する形で公開すること。
※鑑賞用の作品として配布・販売していただくことは問題ございません。
```

* En cuanto a Amitaro, el cambiador de voz en tiempo real, se adhiere a los siguientes términos de uso del Taller de voz de Amitaro. Para más detalles, [aquí](https://amitaro.net/voice/faq/#index_id6)

```
あみたろの声素材やコーパス読み上げ音声を使って音声モデルを作ったり、ボイスチェンジャーや声質変換などを使用して、自分の声をあみたろの声に変換して使うのもOKです。

ただしその場合は絶対に、あみたろ（もしくは小春音アミ）の声に声質変換していることを明記し、あみたろ（および小春音アミ）が話しているわけではないことが誰でもわかるようにしてください。
また、あみたろの声で話す内容は声素材の利用規約の範囲内のみとし、センシティブな発言などはしないでください。
```

* En cuanto a Koto Mahiro, el cambiador de voz en tiempo real, se adhiere a los términos de uso de Replikador. Para más detalles, [aquí](https://kikyohiroto1227.wixsite.com/kikoto-utau/ter%EF%BD%8Ds-of-service)

## Descargo de responsabilidad

No nos hacemos responsables de ningún daño directo, indirecto, consecuente, resultante o especial que surja del uso o la imposibilidad de uso de este software.
