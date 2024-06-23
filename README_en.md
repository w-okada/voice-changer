## VC Client

[Japanese](/README.md) [Korean](/README_ko.md)

## What's New!
- v.2.0.24-alpha Colab version released. ⇒ [Here](./w_okada's_Voice_Changer_version_2_x.ipynb)
- v.2.0.24-alpha
  - Bugfix:
    - Addressed the issue where sound stops when switching modes
  - Others:
    - Enhanced logger
    - Improved error screen
- v.2.0.23-alpha
  - Reorganizing Editions
    - win_std: For typical Windows users. Hardware acceleration via DirectML is available for both ONNX and torch models.
    - win_cuda: For Nvidia GPU owners. Hardware acceleration via CUDA is available for both ONNX and torch models. Requires CUDA 12.4 or later.
    - mac: For Apple Silicon (e.g., M1) users.
  - feature
    - Added the capability to adjust the output buffer when operating in client mode
  - bugfix:
    - Fixed the issue of retaining index and icon when exporting RVC's torch model to onnx model
  - Other:
    - Enhanced logger

- v.2.0.20-alpha
  - Support for torch-cuda. See the edition description [here](docs/01_basic_v2.0.z.md).
  - Bugfix:
    - Unified file encoding to UTF-8
    - 
- v.2.0.16-alpha
  - Added support for experimental version of torch-dml. For a description of the edition, refer to [here](docs/01_basic_v2.0.z.md).
  - Bugfix:
    - Fixed the issue where both pth and index files could not be uploaded simultaneously during rvc file upload.
    - 
- v.2.0.13-alpha
  - Added support for onnxruntime-gpu. Release of the CUDA edition.
  - Bugfix:
    - Addressed issues with onnxcrepe
    - Fixed ID selection issue in Beatrice v2 API
  - Others:
    - Enhanced logger

- v.2.0.6-alpha
  - New
    - Now compatible with M1 series Macs.
      - Confirmed to work on M1 MBA (Monterey) and M2 Pro MBP (Ventura).
      - Looking for reports on performance with Sonoma.
  - Bugfix:
    - Fixed a bug where the pitch would revert when selecting a speaker in Beatrice.
  - Others:
    - Enhanced information gathering for debugging purposes.

- v.2.0.5-alpha
  - VCClient has been rebooted as a second version.
  - Major software structure changes have been made to improve extensibility.
  - Providing REST API to facilitate client development by third parties.
  - Edition system has been completely revamped.
    - The Standard Edition (win) runs on ONNX models by default regardless of the presence of a GPU. Please convert Torch models to ONNX models before use. Hardware acceleration is only effective with ONNX models for users with a GPU.
    - The CUDA Edition (win) is optimized specifically for Nvidia GPUs. It offers further speed enhancements compared to the Standard Edition. Hardware acceleration is only effective with ONNX models.
    - Torch models can also be hardware accelerated using PyTorch models.
    - The Mac Edition is for Mac users with Apple Silicon.
    - Linux users or those with knowledge of Python can clone the repository and run it.
  - Currently, only the Standard Edition is available in the Alpha version.

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
