# Voice Changer

## Overview

This is a fork of [w-okada voice changer](https://github.com/w-okada/voice-changer) that performs real-time voice conversion
using various voice conversion algorithms.

> **NOTE**: Currently, this version is only guaranteed to work with Retrieval-based Voice Conversion (RVC).
> Other voice conversion algorithms are not guaranteed to work.

The fork mostly preserves the same user interface and user experience, gradually improving the existing features and peformance.

The following videos demonstrate how the voice changer works and performs with AMD graphics cards (including integrated GPU!):

[Amd iGPU.webm](https://github.com/deiteris/voice-changer/assets/6103913/af72e31a-cf66-447b-87c9-43f13355402f)

[Amd Dgpu Rx6600m.webm](https://github.com/deiteris/voice-changer/assets/6103913/bc5180b8-3f67-478b-9ea8-c8ae5e59a4c2)

And this one demonstrates how the voice changer works and performs with Nvidia GeForce GTX 1650 laptop:

[Nvidia Dgpu Gtx1650.webm](https://github.com/deiteris/voice-changer/assets/6103913/cb5b0a5b-a96c-460e-9f57-f82bed84559e)

## Supported operated systems

> **NOTE**: macOS builds are not included yet. But you can run from source.

* Windows 10 or later.
* Linux.
* macOS.

## System requirements

> **NOTE**: Minimum requirement means that you will be able to run **ONLY** the voice changer. Voice conversion and gaming at the same time will not provide satisfying experience.

### CPU-only voice conversion

Minimum requirement: Intel Core i5-4690K or AMD FX-6300.

Recommended requirement: Intel Core i5-10400F or AMD Ryzen 5 1600X.

### GPU voice conversion

Minimum requirement:

* An integrated graphics card: AMD Radeon Vega 7 (with AMD Ryzen 5 5600G) or later.
* A dedicated graphics card: Nvidia GeForce GTX 900 Series or later or AMD Radeon RX 500 series or later.

Recommended requirement:

A dedicated graphics card Nvidia GeForce RTX 20 Series or later or AMD Radeon RX 6000 series or later.

## Known issues

### General

* Mozilla Firefox ESR may not display audio devices.

### DirectML (dml) version

* When changing **Chunk**, **Extra** or **Crossfade size** settings, you must switch device to CPU then back to your GPU.
  Otherwise, performance issues can be observed.

* Only `rmvpe_onnx`, `crepe_tiny_onnx` and `crepe_full_onnx` are available in the list of **F0 Det.**.

### All versions

* Export to ONNX does not work.

## How to use

### Running locally

> **NOTE**: These instructions are for Windows only.

#### Before you start

1. [If not installed] Download and install [7-Zip](https://www.7-zip.org/).

1. [If not installed] Download and install [VAC Lite by Muzychenko](https://software.muzychenko.net/freeware/vac470lite.zip).

1. Navigate to the [releases section](https://github.com/deiteris/voice-changer/releases).

#### Check your hardware

1. Open **Task Manager** > **Performance**.

1. Click **CPU**, check and note the processor model on the right. An example: AMD Ryzen 7 5800H with Radeon Graphics.

1. Check and note graphics card models under **GPU**. An example:

   * GPU 0: AMD Radeon RX 6600M.

   * GPU 1: AMD Radeon(TM) Graphics.

#### For AMD/Intel/CPU users

1. Download the `voice-changer-windows-amd64-dml.zip.001` ZIP file.

1. Right-click the ZIP file. In the opened action menu select **7-Zip** > **Extract to "voice-changer-windows-amd64-dml\\"**

#### For Nvidia users

1. Download the `voice-changer-windows-amd64-cuda.zip.001` and `voice-changer-windows-amd64-cuda.zip.002` ZIP files and place them in the same folder.

1. Right-click either of ZIP files. In the opened action menu select **7-Zip** > **Extract to "voice-changer-windows-amd64-cuda\\"**

#### Running

1. Open the extracted folder (`voice-changer-windows-amd64-dml` or `voice-changer-windows-amd64-cuda`) > `dist` > `MMVCServerSIO`.

1. Run `MMVCServerSIO.exe`.

When running the voice changer for the first time, it will start downloading necessary files. Do not close the window until the download finishes.

Once the download is finished, the voice changer will open the user interface using your default web browser.

### Running on Colab/Kaggle

Refer to corresponding [Colab](https://github.com/deiteris/voice-changer/blob/master-custom/Colab_RealtimeVoiceChanger.ipynb) or [Kaggle](https://github.com/deiteris/voice-changer/blob/master-custom/Kaggle_RealtimeVoiceChanger.ipynb) notebooks in this repository and follow their instructions.

## Troubleshooting

> **NOTE**: When any issue with the voice changer occurs, check the command line window (the one that opens during the start) for errors.

### Weight failed to pass verification check

Either the download was incomplete during the first-time start or your files were corrupted. The error will show which files are affected above:

```
Corrupted file pretrain/crepe_onnx_full.onnx: calculated hash 67f6432087eec1887bfcfc6e4045dcae, expected hash e9bb11eb5d3557805715077b30aefebc
Corrupted file pretrain/content_vec_500.onnx: calculated hash ef1f3a8da54c6c7d1ebc708b5824e155, expected hash ab288ca5b540a4a15909a40edf875d1e
Corrupted file pretrain/rmvpe.pt: calculated hash a014255b0460e3cc20c576c01d5583ff, expected hash 7989809b6b54fb33653818e357bcb643
Corrupted file pretrain/rmvpe.onnx: calculated hash 9737c9c9b5ce93bd797a643613ac87e1, expected hash b6979bf69503f8ec48c135000028a7b0
```

Delete the mentioned files and restart the voice changer. Deleted files will be re-downloaded.

### Audio devices are not displayed

1. Make sure that you have given the permission to access the microphone.

1. If you are using Mozilla Firefox ESR, there may be an issue with audio devices. Use other web browser (preferably Chrome or Chromium-based).

### No sound after start

1. Make sure you have selected correct input and output audio devices.

1. Make sure your input device is not muted. Check the microphone volume in the system settings or hardware switch on your headset (usually a button, if present).

### Hearing non-converted voice

In the voice changer, make sure **passthru** is not on (indicated by blinking red color). Click it to switch it off (indicated by solid green color).

### Hearing audio crackles

1. Make sure you are using **VAC by Muzychenko** (indicated by the **Line 1** audio device name).

1. Make sure the **perf** time is smaller than **buf**. Increase **Chunk** or reduce **Extra** and **Crossfade size**.

## Contribution

At the moment, the fork does not accept any code contributions. However, feel free to report any issues
you encounter during usage.

## Working with the source

### Prerequisites

1. [If not installed] Download and install [Python 3.10](https://www.python.org/downloads/release/python-3108/).

1. Open a command line.

1. Verify your Python version by running the following command:

   ```
   python --version
   Python 3.10.8
   ```

1. Navigate to the `server` folder.

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
