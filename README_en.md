## VC Client

[Japanese](/README_ja.md)

## What's New!

- v.1.5.2.5

  - RVC: Support pitch-less model and rvc-webui model
  - so-vits-svc40: some bugfix

- v.1.5.2.4a

  - Fix: Export ONNX

- v.1.5.2.4

  - multi model selection for RVC
  - communication path is fixed to 48KHz

# What is VC Client

1. This is a client software for performing real-time voice conversion using various Voice Conversion (VC) AI. The supported AI for voice conversion are as follows.

- [MMVC](https://github.com/isletennos/MMVC_Trainer)
- [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)
- [RVC(Retrieval-based-Voice-Conversion)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI)
- [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)

2. Distribute the load by running Voice Changer on a different PC
   The real-time voice changer of this application works on a server-client configuration. By running the MMVC server on a separate PC, you can run it while minimizing the impact on other resource-intensive processes such as gaming commentary.

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

3. Cross-platform compatibility
   Supports Windows, Mac (including Apple Silicon M1), Linux, and Google Colaboratory.

# usage

Details are summarized [here](https://zenn.dev/wok/books/0004_vc-client-v_1_5_1_x).

This is an app for performing voice changes with MMVC and so-vits-svc.

It can be used in three main ways, in order of difficulty:

- Using Google Colaboratory (MMVC only)
- Using a pre-built binary
- Setting up an environment with Docker or Anaconda and using it

For those who are not familiar with this software or MMVC, it is recommended to gradually get used to it from the top.

## (1) Use on Google Colaboratory (MMVC only)

You can run it on Google's machine learning platform, Colaboratory. If you have already used Colaboratory, you do not need to prepare anything as the training of MMVC model has been completed. However, the voice changer may have a large time lag depending on the network environment or the situation of Colaboratory.

- [Simple version](https://github.com/w-okada/voice-changer/blob/master/VoiceChangerDemo_Simple.ipynb): You can run it from Colab without any prior setup.
- [Normal version](https://github.com/w-okada/voice-changer/blob/master/VoiceChangerDemo.ipynb): You can load the model by cooperating with Google Drive.

[Explanation video](https://youtu.be/TogfMzXH1T0)

## (2) Usage with pre-built binaries

You can download and run executable binaries.
We offer Windows and Mac versions.

- For Windows user, after unzipping the downloaded zip file, please run the `start_http.bat` file corresponding to your VC.

- For Mac version, after unzipping the downloaded file, double-click the `startHttp.command` file corresponding to your VC. If a message indicating that the developer cannot be verified is displayed, please press the control key and click to run it again (or right-click to run it).

- If you are connecting remotely, please use the `.command` file (Mac) or `.bat` file (Windows) with https instead of http.

- Tsukuyomi-chan, Ami-taro, Kogane Mahiro, and Kogane Kaigetsu require the Content Vec model for their actions. Please download the ContentVec_legacy 500 model from [this repository](https://github.com/auspicious3000/contentvec) and place it in the same folder as startHttp.command or start_http.bat to execute it.

- For the operation of RVC (Retrieval-based-Voice-Conversion) on so-vits-svc 4.0/so-vits-svc 4.0v2, a model of hubert is required. Please download `hubert_base.pt` from [this repository](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main) and store it in the folder where the batch files are located.

- To run DDSP-SVC, you need to download the hubert-soft and enhancer models. Download hubert-soft from [this link](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt) and store it in the folder with the batch files. Download nsf_hifigan_20221211.zip from [this site](https://github.com/openvpi/vocoders/releases/tag/nsf-hifigan-v1) for enhancer. After unzipping, store the nsf_hifigan folder in the folder with the batch files.

- The encoder of DDPS-SVC only supports hubert-soft.

- Please refer to [here](tutorials/tutorial_rvc_ja.md) for the description of each item of GUI to be used in RVC.

| Version   | OS  | フレームワーク                    | link                                                                                       | サポート VC                                                                   | サイズ |
| --------- | --- | --------------------------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- | ------ |
| v.1.5.2.5 | mac | ONNX(cpu), PyTorch(cpu,mps)       | [normal](https://drive.google.com/uc?id=1MNsrVEluBqw_JhSjf_H6H8TNL0wEkg-S&export=download) | MMVC v.1.5.x, MMVC v.1.3.x, so-vits-svc 4.0, RVC                              | 795MB  |
|           | win | ONNX(cpu,cuda), PyTorch(cpu,cuda) | [normal](https://drive.google.com/uc?id=1LILZYbABedhDK6fACOqwERsIdrpIr3gZ&export=download) | MMVC v.1.5.x, MMVC v.1.3.x, so-vits-svc 4.0, so-vits-svc 4.0v2, RVC, DDSP-SVC | 2871MB |

\*\*\* [hugging_face](https://huggingface.co/wok000/vcclient/tree/main) (experimental)

| Version    | OS                                    | Framework | link                                                                                               | VC Support | Size  |
| ---------- | ------------------------------------- | --------- | -------------------------------------------------------------------------------------------------- | ---------- | ----- |
| v.1.5.1.14 | <span style="color: red;">mac</span>  | -         | [Tsukuyomi-chan](https://drive.google.com/uc?id=1ysTt0xOkWFSxyXfFm2GD7i5o22jce4B8&export=download) | -          | 874MB |
|            | <span style="color: red;">mac</span>  | -         | [Kikoto Mahiro](https://drive.google.com/uc?id=1uZW-PSHttQuGXZf9vU7ZGufbYl-nIRs6&export=download)  | -          | 872MB |
|            | <span style="color: red;">mac</span>  | -         | [Amitaro](https://drive.google.com/uc?id=1jc6YXcvt0_z1GezKSvqHQPYFmtZU2KaV&export=download)        | -          | 872MB |
|            | <span style="color: red;">mac</span>  | -         | [Kikoto Kurage](https://drive.google.com/uc?id=1ruaTdhrIJVdz__sDwZEeovzwxrk2ufLT&export=download)  | -          | 873MB |
|            | <span style="color: blue;">win</span> | -         | [Tsukuyomi-chan](https://drive.google.com/uc?id=1QdeotmYP6nnoZt438kB8wvFbYF-C0bhq&export=download) | -          | 823MB |
|            | <span style="color: blue;">win</span> | -         | [Kikoto Mahiro](https://drive.google.com/uc?id=1IJJQj6CHcbyvTwZ5LF6GZSk7FLs5OK6o&export=download)  | -          | 821MB |
|            | <span style="color: blue;">win</span> | -         | [Kikoto Kurage](https://drive.google.com/uc?id=1fiymPcoYzwE1yxyIfC_FTPiFfGEC2jA8&export=download)  | -          | 823MB |
|            | <span style="color: blue;">win</span> | -         | [Amitaro](https://drive.google.com/uc?id=1Vt4WBEOAz0EhIWs3ZRFIcg7ELtSHnYfe&export=download)        | -          | 821MB |

\*1 Tsukuyo Michan uses free character "Tsukuyo Michan" voice data that is publicly available for free. (Details such as terms of use are at the end of the document)

\*2 If unpacking or starting is slow, there is a possibility that virus checking is running on your antivirus software. Please try running it with the file or folder excluded from the target. (At your own risk)

## (2-3) Usage after setting up the environment such as Docker or Anaconda

Clone this repository and use it. Setting up WSL2 is essential for Windows. Additionally, setting up virtual environments such as Docker or Anaconda on WSL2 is also required. On Mac, setting up Python virtual environments such as Anaconda is necessary. Although preparation is required, this method works the fastest in many environments. **<font color="red"> Even without a GPU, it may work well enough with a reasonably new CPU </font>(refer to the section on real-time performance below)**.

[Explanation video on installing WSL2 and Docker](https://youtu.be/POo_Cg0eFMU)

[Explanation video on installing WSL2 and Anaconda](https://youtu.be/fba9Zhsukqw)

To run docker, see [start docker](docker_vcclient/README_en.md).

To run on Anaconda venv, see [server developer's guide](README_dev_en.md)

# Real-time performance

Conversion is almost instantaneous when using GPU.

https://twitter.com/DannadoriYellow/status/1613483372579545088?s=20&t=7CLD79h1F3dfKiTb7M8RUQ

Even with CPU, recent ones can perform conversions at a reasonable speed.

https://twitter.com/DannadoriYellow/status/1613553862773997569?s=20&t=7CLD79h1F3dfKiTb7M8RUQ

With an old CPU (i7-4770), it takes about 1000 msec for conversion.

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
