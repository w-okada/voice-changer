## VC Client

[Japanese](/README.md) [Korean](/README_ko.md)

## What's New!
- Beatrice V2 Training Code Released!!!
  - [Training Code Repository](https://huggingface.co/fierce-cats/beatrice-trainer)
  - [Colab Version](https://github.com/w-okada/beatrice-trainer-colab)
- v.2.0.50-alpha
  - improve:
    - Improved performance in client mode
    - Bundled native client with Mac edition
  - bugfix:
    - Fixed exception when changing input device during BeatriceV2 conversion (if the sampling rate is different, it crashes)
    - Fixed issue where sound gets corrupted when switching from server mode to client mode
- v.2.0.47-alpha
  - feature:
    - Expanded extra frame
  - bugfix:
    - Changed the default speaker ID for Beatrice
    - Fixed errors when model file names are too long
    - Handled situation when monitor device is set to none.
- v.2.0.45-alpha
  - bugfix
    - volume control

# What is VC Client

1. This is a client software for performing real-time voice conversion using various Voice Conversion (VC) AI. The supported AI for voice conversion are as follows.

- [RVC(Retrieval-based-Voice-Conversion)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI)
- [Beatrice JVS Corpus Edition](https://prj-beatrice.com/) * experimental,  (***NOT MIT Licnsence*** see [readme](https://github.com/w-okada/voice-changer/blob/master/server/voice_changer/Beatrice/)) *  Only for Windows, CPU dependent

1. Distribute the load by running Voice Changer on a different PC
   The real-time voice changer of this application works on a server-client configuration. By running the MMVC server on a separate PC, you can run it while minimizing the impact on other resource-intensive processes such as gaming commentary.

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

1. Cross-platform compatibility
   Supports Windows, Mac (including Apple Silicon M1), Linux, and Google Colaboratory.

1. We provide a REST API.

- You can operate it using HTTP clients that are built into the OS, such as curl.
- This allows you to easily achieve the following:
  - Users can register processes that call the REST API in shortcuts, such as in .bat files.
  - Create simple clients to operate remotely.
  - And more.

# Download
Please download it from [Hugging Face](https://huggingface.co/wok000/vcclient000/tree/main).

# Manual

[Manual](docs/01_basic_v2.0.z.md)

# Software Signing

This software is not signed by the developer. A warning message will appear, but you can run the software by clicking the icon while holding down the control key. This is due to Apple's security policy. Running the software is at your own risk.

![image](https://user-images.githubusercontent.com/48346627/212567711-c4a8d599-e24c-4fa3-8145-a5df7211f023.png)

https://user-images.githubusercontent.com/48346627/212569645-e30b7f4e-079d-4504-8cf8-7816c5f40b00.mp4

# Acknowledgments

- [Tachizunda-mon materials](https://seiga.nicovideo.jp/seiga/im10792934)
- [Irasutoya](https://www.irasutoya.com/)
- [Tsukuyomi-chan](https://tyc.rei-yumesaki.net)

> This software uses the voice data of the free material character "Tsukuyomi-chan," which is provided for free by CV. Yumesaki Rei.
>
> - Tsukuyomi-chan Corpus (CV. Yumesaki Rei)
>
> https://tyc.rei-yumesaki.net/material/corpus/
>
> Copyright. Rei Yumesaki

- [Amitaro's Onsozai kobo](https://amitaro.net/)
- [Replica doll](https://kikyohiroto1227.wixsite.com/kikoto-utau)

# Terms of Use

In accordance with the Tsukuyomi-chan Corpus Terms of Use for the Tsukuyomi-chan Real-time Voice Changer, the use of the converted voice for the following purposes is prohibited.

- Criticizing or attacking individuals (the definition of "criticizing or attacking" is based on the Tsukuyomi-chan character license).

- Advocating for or opposing specific political positions, religions, or ideologies.

- Publicly displaying strongly stimulating expressions without proper zoning.

- Publicly disclosing secondary use (use as materials) for others.
  (Distributing or selling as a work for viewing is not a problem.)

Regarding the Real-time Voice Changer Amitaro, we prohibit the following uses in accordance with the terms of use of the Amitaro's koe-sozai kobo.[detail](https://amitaro.net/voice/faq/#index_id6)

Regarding the Real-time Voice Changer Kikoto Mahiro, we prohibit the following uses in accordance with the terms of use of Replica doll.[detail](https://kikyohiroto1227.wixsite.com/kikoto-utau/ter%EF%BD%8Ds-of-service)

# Disclaimer

We are not liable for any direct, indirect, consequential, incidental, or special damages arising out of or in any way connected with the use or inability to use this software.
