[اليابانية](/README.md) /
[الإنجليزية](/docs_i18n/README_en.md) /
[الكورية](/docs_i18n/README_ko.md)/
[الصينية](/docs_i18n/README_zh.md)/
[الألمانية](/docs_i18n/README_de.md)/
[العربية](/docs_i18n/README_ar.md)/
[اليونانية](/docs_i18n/README_el.md)/
[الإسبانية](/docs_i18n/README_es.md)/
[الفرنسية](/docs_i18n/README_fr.md)/
[الإيطالية](/docs_i18n/README_it.md)/
[اللاتينية](/docs_i18n/README_la.md)/
[الماليزية](/docs_i18n/README_ms.md)/
[الروسية](/docs_i18n/README_ru.md)
*جميع اللغات باستثناء اليابانية مترجمة آليًا.

## VCClient

VCClient هو برنامج يقوم بتحويل الصوت في الوقت الحقيقي باستخدام الذكاء الاصطناعي.

## ما الجديد!

* v.2.0.73-beta
  * ميزة جديدة:
    * تحميل نموذج beatrice المعدل
  * إصلاح الأخطاء:
    * تم إصلاح خطأ عدم انعكاس النغمة والصيغة في beatrice v2
    * تم إصلاح خطأ ع��م إمكانية إنشاء ONNX للنماذج التي تستخدم embedder Applio
* v.2.0.72-beta (غير موصى به لوجود أخطاء. تم الإصلاح في v.2.0.73)
  * ميزة جديدة
    * واجهة تحرير Beatrice
    * تخزين النغمة والصيغة لكل صوت من Beatrice
    * دعم لغات متعددة في الواجهة
    * دعم embedder لـ Applio
* v.2.0.70-beta (فقط لأجهزة m1 mac)
  * ميزة جديدة:
    * تم دعم Beatrice v2 beta.1 في نسخة M1 Mac من VCClient.
* v.2.0.69-beta (فقط لأجهزة الويندوز)
  * إصلاح الأخطاء:
    * تم إصلاح خطأ يمنع ظهور زر البدء عند حدوث بعض الاستثناءات
    * تم تعديل مخزن الخرج في وضع جهاز الخادم
    * تم إصلاح خطأ يغير معدل العينة عند تغيير الإعدادات أثناء استخدام وضع جهاز الخادم
    * إصلاح خطأ عند استخدام hubert الياباني
  * متفرقات:
    * إضافة فلتر API المضيف في وضع جهاز الخادم (تمييز)

## التنزيل والروابط ذات الصلة

يمكن تنزيل نسخة الويندوز ونسخة M1 Mac من مستودع hugging face.

* [مستودع VCClient](https://huggingface.co/wok000/vcclient000/tree/main)
* [مستودع Light VCClient لـ Beatrice v2](https://huggingface.co/wok000/light_vcclient_beatrice/tree/main)

*1 بالنسبة للينكس، يرجى استنساخ المستودع لاستخدامه.

### روابط ذات صلة

* [مستودع كود التدريب لـ Beatrice V2](https://huggingface.co/fierce-cats/beatrice-trainer)
* [نسخة Colab من كود التدريب لـ Beatrice V2](https://github.com/w-okada/beatrice-trainer-colab)

### البرامج ذات الصلة

* [مغير الصوت في الوقت الحقيقي VCClient](https://github.com/w-okada/voice-changer)
* [برنامج قراءة النصوص TTSClient](https://github.com/w-okada/ttsclient)
* [برنامج التعرف على الصوت في الوقت الحقيقي ASRClient](https://github.com/w-okada/asrclient)

## ميزات VC Client

## يدعم نماذج الذكاء الاصطناعي المتنوعة

| نماذج الذكاء الاصطناعي                                                                                                     | v.2       | v.1                  | الترخيص                                                                                 |
| ------------------------------------------------------------------------------------------------------------ | --------- | -------------------- | ------------------------------------------------------------------------------------------ |
| [RVC ](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/jp/README.ja.md) | مدعوم | مدعوم            | يرجى الرجوع إلى المستودع.                                                             |
| [Beatrice v1](https://prj-beatrice.com/)                                                                     | غير متاح       | مدعوم (فقط للويندوز) | [خاص](https://github.com/w-okada/voice-changer/tree/master/server/voice_changer/Beatrice) |
| [Beatrice v2](https://prj-beatrice.com/)                                                                     | مدعوم | غير متاح                  | [خاص](https://huggingface.co/wok000/vcclient_model/blob/main/beatrice_v2_beta/readme.md)  |
| [MMVC](https://github.com/isletennos/MMVC_Trainer)                                                           | غير متاح       | مدعوم            | يرجى الرجوع إلى المستودع.                                                             |
| [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)                                               | غير متاح       | مدعوم            | يرجى الرجوع إلى المستودع.                                                             |
| [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)                                                               | غير متاح       | مدعوم            | يرجى الرجوع إلى المستودع.                                                             |

## يدعم كلا من التكوين المستقل وعبر الشبكة

يدعم تحويل الصوت المكتمل على جهاز الكمبيوتر المحلي وكذلك عبر الشبكة.
عند استخدامه عبر الشبكة، يمكن تفريغ عبء تحويل الصوت إلى الخارج عند استخدامه مع تطبيقات عالية التحميل مثل الألعاب.

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

## يدعم منصات متعددة

ويندوز، ماك (M1)، ��ينكس، جوجل كولاب

*1 بالنسبة للينكس، يرجى استنساخ المستودع لاستخدامه.

## يوفر REST API

يمكنك إنشاء عميل باستخدام لغات البرمجة المختلفة.

يمكنك أيضًا استخدام عملاء HTTP المدمجة في نظام التشغيل مثل curl للتحكم.

## استكشاف الأخطاء وإصلاحها

[قسم الاتصال](tutorials/trouble_shoot_communication_ja.md)

## حول توقيع المطور

هذا البرنامج غير موقع من قبل المطور. ستظهر تحذيرات كما هو موضح أدناه، ولكن يمكنك تشغيله بالضغط على مفتاح التحكم أثناء النقر على الأيقونة. هذا بسبب سياسة أمان Apple. التشغيل يكون على مسؤوليتك الخاصة.

![image](https://user-images.githubusercontent.com/48346627/212567711-c4a8d599-e24c-4fa3-8145-a5df7211f023.png)

## الشكر والتقدير

* [مواد Tachi Zundamon](https://seiga.nicovideo.jp/seiga/im10792934)
* [إيراستويا](https://www.irasutoya.com/)
* [Tsukuyomi-chan](https://tyc.rei-yumesaki.net/)

```
  本ソフトウェアの音声合成には、フリー素材キャラクター「つくよみちゃん」が無料公開している音声データを使用しています。
  ■つくよみちゃんコーパス（CV.夢前黎）
  https://tyc.rei-yumesaki.net/material/corpus/
  © Rei Yumesaki
```

* [ورشة عمل صوت Amitaro](https://amitaro.net/)
* [Replikadoru](https://kikyohiroto1227.wixsite.com/kikoto-utau)

## شروط الاستخدام

* بالنسبة لمغير الصوت في الوقت الحقيقي Tsukuyomi-chan، يُحظر استخدام الصوت المحول للأغراض التالية وفقًا لشروط استخدام كوربوس Tsukuyomi-chan.

```

■人を批判・攻撃すること。（「批判・攻撃」の定義は、つくよみちゃんキャラクターライセンスに準じます）

■特定の政治的立場・宗教・思想への賛同または反対を呼びかけること。

■刺激の強い表現をゾーニングなしで公開すること。

■他者に対して二次利用（素材としての利用）を許可する形で公開すること。
※鑑賞用の作品として配布・販売していただくことは問題ございません。
```

* بال��سبة لمغير الصوت في الوقت الحقيقي Amitaro، يُتبع شروط استخدام ورشة عمل صوت Amitaro. التفاصيل[هنا](https://amitaro.net/voice/faq/#index_id6)

```
あみたろの声素材やコーパス読み上げ音声を使って音声モデルを作ったり、ボイスチェンジャーや声質変換などを使用して、自分の声をあみたろの声に変換して使うのもOKです。

ただしその場合は絶対に、あみたろ（もしくは小春音アミ）の声に声質変換していることを明記し、あみたろ（および小春音アミ）が話しているわけではないことが誰でもわかるようにしてください。
また、あみたろの声で話す内容は声素材の利用規約の範囲内のみとし、センシティブな発言などはしないでください。
```

* بالنسبة لمغير الصوت في الوقت الحقيقي Kogane Mahiro، يُتبع شروط استخدام Replikadoru. التفاصيل[هنا](https://kikyohiroto1227.wixsite.com/kikoto-utau/ter%EF%BD%8Ds-of-service)

## إخلاء المسؤولية

لا نتحمل أي مسؤولية عن أي أضرار مباشرة أو غير مباشرة أو تبعية أو خاصة تنشأ عن استخدام أو عدم القدرة على استخدام هذا البرنامج.
