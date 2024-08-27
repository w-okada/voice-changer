## VC Client

[Japanese](/README_ja.md) [Korean](/README_ko.md)

## What's New!
- We have released a sister product, the Text To Speech client.
  - You can enjoy voice generation with a simple interface.
  - For more details, click [here](https://github.com/w-okada/ttsclient).
- Beatrice V2 Training Code Released!!!
  - [Training Code Repository](https://huggingface.co/fierce-cats/beatrice-trainer)
  - [Colab Version](https://github.com/w-okada/beatrice-trainer-colab)
- v.2.0.61-alpha
  - [HERE](https://github.com/w-okada/voice-changer/tree/v.2)
  - feature:
    - You can now specify the crossfade duration.
  - bugfix:
    - Fixed an issue where the non-used elements of a model would still affect performance during model merging by setting their values to zero.
- v.2.0.60-alpha
  - feature:
    - [darkmode](https://github.com/w-okada/voice-changer/issues/1306)
    - [re-introduce pytorch rmvpe](https://github.com/w-okada/voice-changer/issues/1319)
    - [wasapi Exclusive Mode Select](https://github.com/w-okada/voice-changer/issues/1305)
- v.2.0.58-alpha
  - [HERE](https://github.com/w-okada/voice-changer/tree/v.2)
  - feature:
    - SIO Broadcasting
    - Embed ngrok (experimental)
  - improve:
    - Tuning for mobile phones.
  - bugfix:
    - CUI message garbled text on macOS
- v.2.0.55-alpha
  - [HERE](https://github.com/w-okada/voice-changer/tree/v.2)
  - improve:
    - Reduced CPU load of RVC
    - WebSocket support
  - change:
    - Enable no_cui option in startup batch

# What is VC Client

1. This is a client software for performing real-time voice conversion using various Voice Conversion (VC) AI. The supported AI for voice conversion are as follows.

- [MMVC](https://github.com/isletennos/MMVC_Trainer) (only v1)
- [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) (only v1)
- [RVC(Retrieval-based-Voice-Conversion)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI)
- [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) (only v1)
- [Beatrice JVS Corpus Edition](https://prj-beatrice.com/) * experimental,  (***NOT MIT License*** see [readme](https://github.com/w-okada/voice-changer/blob/master/server/voice_changer/Beatrice/)) *  Only for Windows, CPU dependent (only v1)
  - [Beatrice v2](https://prj-beatrice.com/) (only for v2)

1. Distribute the load by running Voice Changer on a different PC
   The real-time voice changer of this application works on a server-client configuration. By running the MMVC server on a separate PC, you can run it while minimizing the impact on other resource-intensive processes such as gaming commentary.

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

3. Cross-platform compatibility
   Supports Windows, Mac (including Apple Silicon M1), Linux, and Google Colaboratory.

# usage

This is an app for performing voice changes with MMVC and so-vits-svc.

It can be used in two main ways, in order of difficulty:

- Using a pre-built binary
- Setting up an environment with Docker or Anaconda and using it

## (1) Usage with pre-built binaries

- You can download and run executable binaries.

- Please see [here](tutorials/tutorial_rvc_en_latest.md) for the tutorial. ([trouble shoot](https://github.com/w-okada/voice-changer/blob/master/tutorials/trouble_shoot_communication_ja.md))

- It's now easy to try it out on [Google Colaboratory](https://github.com/w-okada/voice-changer/tree/v.2/w_okada's_Voice_Changer_version_2_x.ipynb) (requires a ngrok account). You can launch it from the 'Open in Colab' button in the top left corner.

<img src="https://github.com/w-okada/voice-changer/assets/48346627/3f092e2d-6834-42f6-bbfd-7d389111604e" width="400" height="150">

- We offer Windows and Mac versions on [hugging face](https://huggingface.co/wok000/vcclient000/tree/main)
- v2 for Windows
  - Please download and use `vcclient_win_std_xxx.zip`. You can perform voice conversion using a reasonably high-performance CPU without a GPU, or by utilizing DirectML to leverage GPUs (AMD, Nvidia). v2 supports both torch and onnx.
  - If you have an Nvidia GPU, you can achieve faster voice conversion by using `vcclient_win_cuda_xxx.zip`.
- v2 for Mac (Apple Silicon)
  - Please download and use `vcclient_mac_xxx.zip`.
- v1
  - If you are using a Windows and Nvidia GPU, please download ONNX (cpu, cuda), PyTorch (cpu, cuda).
  - If you are using a Windows and AMD/Intel GPU, please download ONNX (cpu, DirectML) and PyTorch (cpu, cuda). AMD/Intel GPUs are only enabled for ONNX models.
  - In either case, for GPU support, PyTorch and Onnxruntime are only enabled if supported.
  - If you are not using a GPU on Windows, please download ONNX (cpu, cuda) and PyTorch (cpu, cuda).

- For Windows user, after unzipping the downloaded zip file, please run the `start_http.bat` file corresponding to your VC.

- For Mac version, after unzipping the downloaded file, double-click the `startHttp.command` file corresponding to your VC. If a message indicating that the developer cannot be verified is displayed, please press the control key and click to run it again (or right-click to run it).

- If you are connecting remotely, please use the `.command` file (Mac) or `.bat` file (Windows) with https instead of http.

- The encoder of DDPS-SVC only supports hubert-soft.

- [Download from hugging face](https://huggingface.co/wok000/vcclient000/tree/main)

## (2) Usage after setting up the environment such as Docker or Anaconda

Clone this repository and use it. Setting up WSL2 is essential for Windows. Additionally, setting up virtual environments such as Docker or Anaconda on WSL2 is also required. On Mac, setting up Python virtual environments such as Anaconda is necessary. Although preparation is required, this method works the fastest in many environments. **<font color="red"> Even without a GPU, it may work well enough with a reasonably new CPU </font>(refer to the section on real-time performance below)**.

[Explanation video on installing WSL2 and Docker](https://youtu.be/POo_Cg0eFMU)

[Explanation video on installing WSL2 and Anaconda](https://youtu.be/fba9Zhsukqw)

To run docker, see [start docker](docker_vcclient/README_en.md).

To run on Anaconda venv, see [server developer's guide](README_dev_en.md)

To run on Linux using an AMD GPU, see [setup guide linux](tutorials/tutorial_anaconda_amd_rocm.md)


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
