[Japonais](/README.md) /
[Anglais](/docs_i18n/README_en.md) /
[Coréen](/docs_i18n/README_ko.md)/
[Chinois](/docs_i18n/README_zh.md)/
[Allemand](/docs_i18n/README_de.md)/
[Arabe](/docs_i18n/README_ar.md)/
[Grec](/docs_i18n/README_el.md)/
[Espagnol](/docs_i18n/README_es.md)/
[Français](/docs_i18n/README_fr.md)/
[Italien](/docs_i18n/README_it.md)/
[Latin](/docs_i18n/README_la.md)/
[Malais](/docs_i18n/README_ms.md)/
[Russe](/docs_i18n/README_ru.md)
*Les langues autres que le japonais sont traduites automatiquement.

## VCClient

VCClient est un logiciel qui utilise l'IA pour effectuer une conversion vocale en temps réel.

## Quoi de neuf !

* v.2.0.73-beta
  * nouvelle fonctionnalité :
    * Téléchargement du modèle Beatrice modifié
  * correction de bug :
    * Correction du bug où le pitch et le formant de Beatrice v2 n'étaient pas appliqués
    * Correction du bug empêchant la création de l'ONNX pour les modèles utilisant l'embedder d'Applio
* v.2.0.72-beta (non recommandé en raison de bugs, corrigé dans v.2.0.73)
  * nouvelle fonctionnalité
    * GUI d'édition de Beatrice
    * Mémorisation du pitch et du formant pour chaque voix de Beatrice
    * GUI multilingue
    * Support de l'embedder Applio
* v.2.0.70-beta (uniquement pour Mac M1)
  * nouvelle fonctionnalité :
    * VCClient pour Mac M1 prend désormais en charge Beatrice v2 beta.1.
* v.2.0.69-beta (uniquement pour Windows)
  * correction de bug :
    * Correction d'un bug où le bouton de démarrage ne s'affichait pas lors de certaines exceptions
    * Ajustement du tampon de sortie en mode appareil serveur
    * Correction d'un bug où le taux d'échantillonnage changeait lors de la modification des paramètres en mode appareil serveur
    * Correction de bug lors de l'utilisation de hubert japonais
  * divers :
    * Ajout d'un filtre API hôte en mode appareil serveur (mise en évidence)

## Téléchargement et liens associés

Les versions Windows et Mac M1 peuvent être téléchargées depuis le référentiel hugging face.

* [Référentiel de VCClient](https://huggingface.co/wok000/vcclient000/tree/main)
* [Référentiel de Light VCClient pour Beatrice v2](https://huggingface.co/wok000/light_vcclient_beatrice/tree/main)

*1 Pour Linux, veuillez cloner le référentiel pour l'utiliser.

### Liens associés

* [Référentiel de code d'entraînement Beatrice V2](https://huggingface.co/fierce-cats/beatrice-trainer)
* [Version Colab du code d'entraînement Beatrice V2](https://github.com/w-okada/beatrice-trainer-colab)

### Logiciels associés

* [Changeur de voix en temps réel VCClient](https://github.com/w-okada/voice-changer)
* [Logiciel de synthèse vocale TTSClient](https://github.com/w-okada/ttsclient)
* [Logiciel de reconnaissance vocale en temps réel ASRClient](https://github.com/w-okada/asrclient)

## Caractéristiques de VC Client

## Prend en charge divers modèles d'IA

| Modèle d'IA                                                                                                     | v.2       | v.1                  | Licence                                                                                 |
| ------------------------------------------------------------------------------------------------------------ | --------- | -------------------- | ------------------------------------------------------------------------------------------ |
| [RVC ](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/jp/README.ja.md) | pris en charge | pris en charge            | Veuillez consulter le référentiel.                                                             |
| [Beatrice v1](https://prj-beatrice.com/)                                                                     | n/a       | pris en charge (uniquement Windows) | [Propriétaire](https://github.com/w-okada/voice-changer/tree/master/server/voice_changer/Beatrice) |
| [Beatrice v2](https://prj-beatrice.com/)                                                                     | pris en charge | n/a                  | [Propriétaire](https://huggingface.co/wok000/vcclient_model/blob/main/beatrice_v2_beta/readme.md)  |
| [MMVC](https://github.com/isletennos/MMVC_Trainer)                                                           | n/a       | pris en charge            | Veuillez consulter le référentiel.                                                             |
| [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)                                               | n/a       | pris en charge            | Veuillez consulter le référentiel.                                                             |
| [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)                                                               | n/a       | pris en charge            | Veuillez consulter le référentiel.                                                             |

## Prend en charge les configurations autonomes et via réseau

Prend en charge la conversion vocale entièrement sur PC local ainsi que via réseau.
En utilisant via réseau, la charge de conversion vocale peut être déportée à l'extérieur lors de l'utilisation simultanée avec des applications à forte charge comme les jeux.

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

## Compatible avec plusieurs plateformes

Windows, Mac(M1), Linux, Google Colab

*1 Pour Linux, veuillez cloner le référentiel pour l'utiliser.

## Fournit une API REST

Vous pouvez créer des clients dans divers langages de programmation.

Vous pouvez également utiliser des clients HTTP intégrés au système d'exploitation comme curl pour les opérations.

## Dépannage

[Communication](tutorials/trouble_shoot_communication_ja.md)

## À propos de la signature du développeur

Ce logiciel n'est pas signé par le développeur. Un avertissement s'affiche comme ci-dessous, mais vous pouvez l'exécuter en cliquant sur l'icône tout en maintenant la touche Contrôle. Ceci est dû à la politique de sécurité d'Apple. L'exécution est à vos propres risques.

![image](https://user-images.githubusercontent.com/48346627/212567711-c4a8d599-e24c-4fa3-8145-a5df7211f023.png)

## Remerciements

* [Matériel de Tachi Zundamon](https://seiga.nicovideo.jp/seiga/im10792934)
* [Irasutoya](https://www.irasutoya.com/)
* [Tsukuyomi-chan](https://tyc.rei-yumesaki.net/)

```
  本ソフトウェアの音声合成には、フリー素材キャラクター「つくよみちゃん」が無料公開している音声データを使用しています。
  ■つくよみちゃんコーパス（CV.夢前黎）
  https://tyc.rei-yumesaki.net/material/corpus/
  © Rei Yumesaki
```

* [Atelier de voix d'Amitaro](https://amitaro.net/)
* [Replika Doll](https://kikyohiroto1227.wixsite.com/kikoto-utau)

## Conditions d'utilisation

* En ce qui concerne le changeur de voix en temps réel Tsukuyomi-chan, l'utilisation de la voix convertie est interdite aux fins suivantes, conformément aux conditions d'utilisation du corpus Tsukuyomi-chan.

```

■人を批判・攻撃すること。（「批判・攻撃」の定義は、つくよみちゃんキャラクターライセンスに準じます）

■特定の政治的立場・宗教・思想への賛同または反対を呼びかけること。

■刺激の強い表現をゾーニングなしで公開すること。

■他者に対して二次利用（素材としての利用）を許可する形で公開すること。
※鑑賞用の作品として配布・販売していただくことは問題ございません。
```

* En ce qui concerne le changeur de voix en temps réel Amitaro, il est conforme aux conditions d'utilisation de l'atelier de voix d'Amitaro. Pour plus de détails, [ici](https://amitaro.net/voice/faq/#index_id6)

```
あみたろの声素材やコーパス読み上げ音声を使って音声モデルを作ったり、ボイスチェンジャーや声質変換などを使用して、自分の声をあみたろの声に変換して使うのもOKです。

ただしその場合は絶対に、あみたろ（もしくは小春音アミ）の声に声質変換していることを明記し、あみたろ（および小春音アミ）が話しているわけではないことが誰でもわかるようにしてください。
また、あみたろの声で話す内容は声素材の利用規約の範囲内のみとし、センシティブな発言などはしないでください。
```

* En ce qui concerne le changeur de voix en temps réel Koto Mahiro, il est conforme aux conditions d'utilisation de Replika Doll. Pour plus de détails, [ici](https://kikyohiroto1227.wixsite.com/kikoto-utau/ter%EF%BD%8Ds-of-service)

## Clause de non-responsabilité

Nous déclinons toute responsabilité pour tout dommage direct, indirect, consécutif, résultant ou spécial causé par l'utilisation ou l'incapacité d'utiliser ce logiciel.
