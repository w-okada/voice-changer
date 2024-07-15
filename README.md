# Voice Changer

## Overview

This is a fork of [w-okada voice changer](https://github.com/w-okada/voice-changer) that performs real-time voice conversion
using various voice conversion algorithms.

> **NOTE**: This version works only with Retrieval-based Voice Conversion (RVC).

The fork aims to improve the overall performance for any backend, and at the same time introducing new features and improving
user experience.

The following videos demonstrate how the voice changer works and performs with AMD graphics cards (including integrated GPU!):

[Amd iGPU.webm](https://github.com/deiteris/voice-changer/assets/6103913/67354d4e-51aa-425c-bd23-d33ad2baf824)

[Amd Dgpu Rx6600m.webm](https://github.com/deiteris/voice-changer/assets/6103913/b932296c-36e4-4150-9306-e80b5c7e4afb)

And this one demonstrates how the voice changer works and performs with Nvidia GeForce GTX 1650 laptop:

[Nvidia Dgpu Gtx 1650.webm](https://github.com/deiteris/voice-changer/assets/6103913/3985e6f3-c22d-4899-ad93-057c599b8b53)

## Supported operated systems

* Windows 10 or later.
* Linux.
* macOS 12 Monterey or later. With Apple Silicon or Intel CPU.

## System requirements

> **NOTE**: Minimum requirement means that you will be able to run **ONLY** the voice changer. Voice conversion and gaming at the same time will not provide satisfying experience with minimum requirements in most cases.

RAM: at least 6GB.

Disk space: at least 6GB of free disk space. For fast model loading, SSD is recommended.

### For CPU-only voice conversion

Minimum requirement: Intel Core i5-4690K or AMD FX-6300.

Recommended requirement: Intel Core i5-10400F or AMD Ryzen 5 1600X.

### For GPU voice conversion

Minimum VRAM required: 2GB (in FP32 mode), ~1GB (in FP16 mode, if supported).

Minimum requirement:

* An integrated graphics card: AMD Radeon Vega 7 (with AMD Ryzen 5 5600G) or later.
* A dedicated graphics card: Nvidia GeForce GTX 900 Series or later or AMD Radeon RX 400 series or later.

Recommended requirement:

A dedicated graphics card Nvidia GeForce RTX 20 Series or later or AMD Radeon RX 6000 series or later.

> **NOTE**: Nvidia GPUs prior to GTX 900 Series will not work with CUDA version. However, they may be used with DirectML version.

## Known issues

### General

* Mozilla Firefox ESR may not display audio devices.

### DirectML (dml) version

* When changing **Chunk**, **Extra** or **Crossfade size** settings, you must switch device to CPU then back to your GPU.
  Otherwise, performance issues can be observed.

* Only `rmvpe_onnx`, `fcpe_onnx`, `crepe_tiny_onnx` and `crepe_full_onnx` are available in the list of **F0 Det.**.

* When using a laptop with integrated GPU and dedicated GPU, severely degraded performance (up to 50% reduction) can be observed when running the voice changer on built-in display.

* Slightly degraded performance (up to 25% reduction) can be observed with multi-GPU setups.

* AMD Radeon RX 7000 series may be unable to achieve low latency (below 256ms).

### Nvidia version

* When starting voice conversion for the first time, it may take up to 5-7 seconds to start outputting the converted voice.

### All versions

* Bad voice quality may be observed when changing Chunk with voice conversion started. The workaround is to stop and start conversion again.

* Errors may occur when switching to different device with voice conversion enabled.

## How to use

### Running locally on Windows

#### Before you start

1. [If not installed] Download and install [7-Zip](https://www.7-zip.org/) or [WinRAR](https://www.win-rar.com/download.html).

1. [If not installed] Download and install [VAC Lite by Muzychenko](https://software.muzychenko.net/freeware/vac470lite.zip).

1. Navigate to the [releases section](https://github.com/deiteris/voice-changer/releases).

#### Check your hardware

1. Open **Task Manager** > **Performance**.

1. Click **CPU**, check and note the processor model on the right. An example: AMD Ryzen 7 5800H with Radeon Graphics.

1. Check and note graphics card models under **GPU**. An example:

   * GPU 0: AMD Radeon RX 6600M.

   * GPU 1: AMD Radeon(TM) Graphics.

#### For AMD/Intel/CPU users

> NOTE: For AMD users, the recommended driver version is `24.6.1` or later.

1. Download the `voice-changer-windows-amd64-dml.zip` ZIP file.

1. Right-click the ZIP file. In the opened action menu select **7-Zip** > **Extract to "voice-changer-windows-amd64-dml\\"**.

#### For Nvidia users

1. Make sure your Nvidia driver version is `528.33` or later. [Click here](https://www.nvidia.com/en-gb/drivers/drivers-faq) to learn how to check your driver version.

1. Download the `voice-changer-windows-amd64-cuda.zip.001` and `voice-changer-windows-amd64-cuda.zip.002` ZIP files and place them in the same folder.

1. Right-click the `voice-changer-windows-amd64-cuda.zip.001` ZIP file. In the opened action menu select **7-Zip** > **Extract to "voice-changer-windows-amd64-cuda\\"**. This will unpack **both** files, no need to unpack them separately.

The following examples demonstrate the unpacking process:

* 7-Zip.
  ![unzip_cuda](https://github.com/deiteris/voice-changer/assets/6103913/f33ebb39-b527-462e-bd0c-6007d26aba35)
* WinRAR.
  ![unzip_cuda_winrar](https://github.com/deiteris/voice-changer/assets/6103913/1f8d63db-01b6-427f-9ee9-c674a61d0ecf)

#### Running the voice changer

1. Open the extracted folder (`voice-changer-windows-amd64-dml` or `voice-changer-windows-amd64-cuda`) > `MMVCServerSIO`.

1. Run `MMVCServerSIO.exe`.

When running the voice changer for the first time, it will start downloading necessary files. Do not close the window until the download finishes.

Once the download is finished, the voice changer will open the user interface using your default web browser.

### Running locally on macOS

> **NOTE**: macOS support is experimental.

#### For Apple Silicon (Apple M1, etc.) users

1. Download the `voice-changer-macos-arm64-cpu.tar.gz` file.

1. Double-click the file. The voice changer will unpack and the `MMVCServerSIO` folder will appear.

#### For Intel users

> **NOTE**: The voice changer would work best if your Intel-based machine has AMD graphics. If your machine has only Intel integrated graphics, only CPU will be utilized.

1. Download the `voice-changer-macos-amd64-cpu.tar.gz` file.

1. Double-click the file. The voice changer will unpack and the `MMVCServerSIO` folder will appear.

#### Removing Apple quarantine attribute

> **NOTE**: Currently, this step is mandatory. This may be improved in the future.

1. Open Terminal.

1. Run the following command:

   ```
   xattr -dr com.apple.quarantine <Path to extracted MMVCServerSIO folder>
   ```

   For example, if you extracted the voice changer to your desktop, the command may look as follows:

   ```
   xattr -dr com.apple.quarantine ~/Desktop/MMVCServerSIO
   ```

#### Running the voice changer

1. Open the extracted `MMVCServerSIO` folder.

1. Double-click `MMVCServerSIO` to run the voice changer.

### Running on Colab/Kaggle

Refer to corresponding [Colab](https://github.com/deiteris/voice-changer/blob/master-custom/Colab_RealtimeVoiceChanger.ipynb) or [Kaggle](https://github.com/deiteris/voice-changer/blob/master-custom/Kaggle_RealtimeVoiceChanger.ipynb) notebooks in this repository and follow their instructions.

## Troubleshooting

> **NOTE**: When any issue with the voice changer occurs, check the command line window (the one that opens during the start) for errors.

### Exceptions.WeightDownloadException: 'Failed to download weight.'

Either the remote files have changed or your files were corrupted. The error will show which files are affected above the error:

```
[Voice Changer] 'pretrain/content_vec_500.onnx failed to pass hash verification check. Got 1931e237626b80d65ae44cbacd4a5197, expected ab288ca5b540a4a15909a40edf875d1e'
[Voice Changer] 'pretrain/rmvpe.onnx failed to pass hash verification check. Got 65030149d579a65f15aa7e85769c32f1, expected b6979bf69503f8ec48c135000028a7b0'
```

Find and delete the mentioned files from the voice changer folder and restart the voice changer. Deleted files will be re-downloaded.

### Audio devices are not displayed

1. Make sure that you have given the permission to access the microphone.

1. If you are using Mozilla Firefox ESR, there may be an issue with audio devices. Use other web browser (preferably Chrome or Chromium-based).

### No sound after start

1. Make sure you have selected correct input and output audio devices.

1. Make sure your input device is not muted. Check the microphone volume in the system settings or hardware switch on your headset (usually a button, if present).

1. Try switching input device to **none**, then back to your microphone.

1. If nothing helped, check the command line that opens when your start the app. In case there are any error messages containing `Exception` or `error`, report an issue and provide full text from the command line.

### Hearing non-converted voice

In the voice changer, make sure **passthru** is not on (indicated by blinking red color). Click it to switch it off (indicated by solid green color).

### Hearing audio crackles

1. Make sure you are using **VAC by Muzychenko** (indicated by the **Line 1** audio device name).

1. Make sure the **perf** time is smaller than **Chunk**. Increase **Chunk** or reduce **Extra** and **Crossfade size**.

## Contribution

At the moment, the fork does not accept any code contributions. However, feel free to report any issues
you encounter during usage.

## Working with the source

### Prerequisites

1. [If not installed] Download and install [Python 3.10](https://www.python.org/downloads/release/python-3108/).

1. [If not installed] Download and install git.

1. Open a command line.

1. Verify your Python version by running the following command:

   ```
   python --version
   Python 3.10.8
   ```

1. Clone the repository.

1. Navigate to the `server` folder.

### Setting up the environment

1. [If not set up] Set up virtual environment with the following command:

   ```
   python -m venv venv
   ```

1. Activate virtual environment using one of the following commands:

   * For Windows:

     ```
     .\venv\Scripts\activate.ps1
     ```

   * For Linux/macOS:

     ```
     source ./venv/bin/activate
     ```

1. Install the requirements using one of the following commands:

   * For AMD/Intel/CPU (Windows only):

     ```
     pip install -r requirements-common.txt -r requirements-dml.txt
     ```

   * For Nvidia (any OS):

     ```
     pip install -r requirements-common.txt -r requirements-cuda.txt
     ```

   * For AMD ROCm (Linux only):

     ```
     pip install -r requirements-common.txt -r requirements-rocm.txt
     ```

   * For CPU (Linux/macOS only):

     ```
     pip install -r requirements-common.txt -r requirements-cpu.txt
     ```

### Running the server

Run the server by executing `main.py`.

```
python ./main.py
```

This will run the server with default settings. Note that it will not open the web browser by default, copy the address from command line.

### Building a package

1. [If not installed] Install `pyinstaller` with the following command:

   ```
   pip install --upgrade pip wheel setuptools pyinstaller
   ```

1. Run the following command to build an executable:

   ```
   pyinstaller --clean -y --dist ./dist --workpath /tmp MMVCServerSIO.spec
   ```

   This will output the resulting executable in the `dist` folder.
