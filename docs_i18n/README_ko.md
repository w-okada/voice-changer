[일본어](/README.md) /
[영어](/docs_i18n/README_en.md) /
[한국어](/docs_i18n/README_ko.md)/
[중국어](/docs_i18n/README_zh.md)/
[독일어](/docs_i18n/README_de.md)/
[아랍어](/docs_i18n/README_ar.md)/
[그리스어](/docs_i18n/README_el.md)/
[스페인어](/docs_i18n/README_es.md)/
[프랑스어](/docs_i18n/README_fr.md)/
[이탈리아어](/docs_i18n/README_it.md)/
[라틴어](/docs_i18n/README_la.md)/
[말레이어](/docs_i18n/README_ms.md)/
[러시아어](/docs_i18n/README_ru.md)
*일본어 외에는 기계 번역입니다.

## VCClient

VCClient는 AI를 사용하여 실시간 음성 변환을 수행하는 소프트웨어입니다.

## What's New!

* v.2.0.78-beta
  * 버그 수정: RVC 모델 업로드 오류 회피
  * ver.1.x와 동시에 실행 가능해졌습니다.
  * 선택 가능한 chunk size를 늘렸습니다.
* v.2.0.77-beta (RTX 5090 전용, 실험적)
  * RTX 5090 관련 모듈 지원 (개발자가 RTX 5090을 보유하지 않아 검증되지 않음)
* v.2.0.76-beta
  * new feature:
    * Beatrice: 화자 병합 구현
    * Beatrice: 자동 피치 시프트
  * bugfix:
    * 서버 모드에서 장치 선택 시의 문제 해결
* v.2.0.73-beta
  * new feature:
    * 편집한 beatrice 모델 다운로드
  * bugfix:
    * beatrice v2의 pitch, formant가 반영되지 않는 버그를 수정
    * Applio의 embedder를 사용하고 있는 모델의 ONNX가 생성되지 않는 버그를 수정

## 다운로드 및 관련 링크

Windows 버전, M1 Mac 버전은 hugging face의 리포지토리에서 다운로드할 수 있습니다.

* [VCClient의 리포지토리](https://huggingface.co/wok000/vcclient000/tree/main)
* [Light VCClient for Beatrice v2의 리포지토리](https://huggingface.co/wok000/light_vcclient_beatrice/tree/main)

*1 Linux는 리포지토리를 클론하여 사용하세요.

### 관련 링크

* [Beatrice V2 트레이닝 코드의 리포지토리](https://huggingface.co/fierce-cats/beatrice-trainer)
* [Beatrice V2 트레이닝 코드 Colab 버전](https://github.com/w-okada/beatrice-trainer-colab)

### 관련 소프트웨어

* [실시간 보이스 체인저 VCClient](https://github.com/w-okada/voice-changer)
* [읽기 소프트웨어 TTSClient](https://github.com/w-okada/ttsclient)
* [실시간 음성 인식 소프트웨어 ASRClient](https://github.com/w-okada/asrclient)

## VC Client의 특징

## 다양한 AI 모델을 지원

| AI 모델                                                                                                     | v.2       | v.1                  | 라이선스                                                                                 |
| ------------------------------------------------------------------------------------------------------------ | --------- | -------------------- | ------------------------------------------------------------------------------------------ |
| [RVC ](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/jp/README.ja.md) | supported | supported            | 리포지토리를 참조하세요.                                                             |
| [Beatrice v1](https://prj-beatrice.com/)                                                                     | n/a       | supported (only win) | [독자](https://github.com/w-okada/voice-changer/tree/master/server/voice_changer/Beatrice) |
| [Beatrice v2](https://prj-beatrice.com/)                                                                     | supported | n/a                  | [독자](https://huggingface.co/wok000/vcclient_model/blob/main/beatrice_v2_beta/readme.md)  |
| [MMVC](https://github.com/isletennos/MMVC_Trainer)                                                           | n/a       | supported            | 리포지토리를 참조하세요.                                                             |
| [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)                                               | n/a       | supported            | 리포지토리를 참조하세요.                                                             |
| [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)                                                               | n/a       | supported            | 리포지토리를 참조하세요.                                                             |

## 독립형, 네트워크 경유의 두 가지 구성을 지원

로컬 PC에서 완료된 음성 변환과 네트워크를 통한 음성 변환을 지원합니다.
네트워크를 통해 사용하면 게임 등 고부하 애플리케이션과 동시에 사용할 때 음성 변환의 부하를 외부로 오프로드할 수 있습니다.

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

## 다중 플랫폼에 대응

Windows, Mac(M1), Linux, Google Colab

*1 Linux는 리포지토리를 클론하여 사용하세요.

## REST API를 제공

각종 프로그래밍 언어로 클라이언트를 만들 수 있습니다.

또한, curl 등 OS에 내장된 HTTP 클라이언트를 사용하여 조작할 수 있습니다.

## 문제 해결

[통신 편](tutorials/trouble_shoot_communication_ja.md)

## 개발자의 서명에 대해

이 소프트웨어는 개발자의 서명이 되어 있지 않습니다. 아래와 같은 경고가 나오지만, 컨트롤 키를 누른 상태에서 아이콘을 클릭하면 실행할 수 있습니다. 이는 Apple의 보안 정책에 따른 것입니다. 실행은 본인의 책임입니다.

![image](https://user-images.githubusercontent.com/48346627/212567711-c4a8d599-e24c-4fa3-8145-a5df7211f023.png)

## Acknowledgments

* [타치준다몬 소재](https://seiga.nicovideo.jp/seiga/im10792934)
* [일러스트야](https://www.irasutoya.com/)
* [츠쿠요미짱](https://tyc.rei-yumesaki.net/)

```
  本ソフトウェアの音声合成には、フリー素材キャラクター「つくよみちゃん」が無料公開している音声データを使用しています。
  ■つくよみちゃんコーパス（CV.夢前黎）
  https://tyc.rei-yumesaki.net/material/corpus/
  © Rei Yumesaki
```

* [아미타로의 목소리 소재 공방](https://amitaro.net/)
* [레플리카돌](https://kikyohiroto1227.wixsite.com/kikoto-utau)

## 이용 약관

* 실시간 보이스 체인저 츠쿠요미짱에 대해서는 츠쿠요미짱 코퍼스의 이용 약관에 따라 다음 목적에서 변환 후 음성을 사용하는 것을 금지합니다.

```

■人を批判・攻撃すること。（「批判・攻撃」の定義は、つくよみちゃんキャラクターライセンスに準じます）

■特定の政治的立場・宗教・思想への賛同または反対を呼びかけること。

■刺激の強い表現をゾーニングなしで公開すること。

■他者に対して二次利用（素材としての利用）を許可する形で公開すること。
※鑑賞用の作品として配布・販売していただくことは問題ございません。
```

* 실시간 보이스 체인저 아미타로에 대해서는 아미타로の목소리 소재 공방의 다음 이용 약관에 따릅니다. 자세한 내용은[여기](https://amitaro.net/voice/faq/#index_id6)

```
あみたろの声素材やコーパス読み上げ音声を使って音声モデルを作ったり、ボイスチェンジャーや声質変換などを使用して、自分の声をあみたろの声に変換して使うのもOKです。

ただしその場合は絶対に、あみたろ（もしくは小春音アミ）の声に声質変換していることを明記し、あみたろ（および小春音アミ）が話しているわけではないことが誰でもわかるようにしてください。
また、あみたろの声で話す内容は声素材の利用規約の範囲内のみとし、センシティブな発言などはしないでください。
```

* 실시간 보이스 체인저 황금 마히로에 대해서는 레플리카돌의 이용 약관에 따릅니다. 자세한 내용은[여기](https://kikyohiroto1227.wixsite.com/kikoto-utau/ter%EF%BD%8Ds-of-service)

## 면책 조항

이 소프트웨어의 사용 또는 사용 불가으로 인해 발생한 어떠한 직접 손해, 간접 손해, 파급적 손해, 결과적 손해 또는 특별 손해에 대해서도 일체 책임을 지지 않습니다.
