# Realtime Voice Changer Client for RVC Tutorial (v.1.5.3.1)

# Introduction

This application is client software for real-time voice conversion that supports various voice conversion models. This document provides a description for voice conversion limited to [RVC(Retrieval-based-Voice-Conversion)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI).

From the following, the original [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) is referred to as the original-RVC, [RVC-WebUI](https://github.com/ddPn08/rvc-webui) created by ddPn08 is referred to as ddPn08-RVC.

## Notes

- Model training must be done separately.
  - If you want to learn by yourself, please go to [original-RVC](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI) or [ddPn08RVC](https://github.com/ddPn08/rvc-webui).
  - [Recording app on Github Pages](https://w-okada.github.io/voice-changer/) is convenient for preparing voice for learning on the browser.
    - [Commentary video] (https://youtu.be/s_GirFEGvaA)
  - [TIPS for training](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/training_tips_en.md) has been published, so please refer to it.

# Steps up to startup

## Start GUI

### Windows version,

Unzip the downloaded zip file and run `start_http.bat`.

### Mac version

After extracting the download file, execute `startHttp.command`. If it shows that the developer cannot be verified, press the control key again and click to execute (or right-click to execute).

### Precautions when connecting remotely

When connecting remotely, please use `.bat` file (win) and `.command` file (mac) where http is replaced with https.

### Console

When you run a .bat file (Windows) or .command file (Mac), a screen like the following will be displayed and various data will be downloaded from the Internet at the initial start-up. Depending on your environment, it may take 1-2 minutes in many cases.

![image](https://github.com/w-okada/voice-changer/assets/48346627/88a30097-2fb3-4c50-8bf1-19c41f27c481)

### GUI

Once the download of required data for launching is complete, a Launcher screen like the following will appear. Please select RVC from this screen.

![v1.5.3.1 RVC](https://github.com/w-okada/voice-changer/assets/48346627/0f407779-7798-49f9-a542-663d80807cdb)

# Quick start

At startup, you can immediately perform voice conversion using the data downloaded.
Select the microphone and speakers in (1) of the figure below, then press the start button in (2). After a few seconds of data loading, the voice conversion will start. For those who are not used to it, it is recommended to select client device in (1) to select the microphone and speakers. (The difference between server device will be described later.)

![image](https://github.com/w-okada/voice-changer/assets/48346627/ce2f8be7-852e-4b78-adce-1df8cad9fbab)

## Configurable items

The items that can be set with the GUI are divided into sections like the figure below. Each section can be opened and closed by clicking the title.

![image](https://github.com/w-okada/voice-changer/assets/48346627/a5eab90c-c0af-42cd-abfb-e897d333d1ff)

## Title

![image](https://github.com/w-okada/rvc-trainer-docker/assets/48346627/0ea2106d-9da9-493b-aee0-8320fa58e273)

Icons are links.

| Icon                                                                                                                               | To                |
| :--------------------------------------------------------------------------------------------------------------------------------- | :---------------- |
| <img src="https://github.com/w-okada/rvc-trainer-docker/assets/48346627/97c18ca5-eee5-4be2-92a7-8092fff960f2" width="32"> Octocat  | github repository |
| <img src="https://github.com/w-okada/rvc-trainer-docker/assets/48346627/751164e4-7b7d-4d7e-b49c-1ad660bf7439" width="32"> question | manual            |
| <img src="https://github.com/w-okada/rvc-trainer-docker/assets/48346627/7bc188db-3aae-43eb-98a1-34aacc16173d" width="32"> spanner  | tools             |
| <img src="https://github.com/w-okada/rvc-trainer-docker/assets/48346627/5db16acc-e901-40d2-8fc2-1fb9fd67f59c" width="32"> coffee   | donation          |

### claer setting

Initialize configuration.

### reload

Reload the window.

### re-select vc

Return to launcher.

## server control

### start

`start` starts the server, `stop` stops the server

### monitor

Indicates the status of real-time conversion.

The lag from voicing to conversion is `buf + res seconds`. Adjust so that the buf time is longer than res.

If you are using the device in server device mode, this display will not be shown. It will be displayed on the console side.

#### vol

This is the volume after voice conversion.

#### buf

It is the length (ms) of one section to cut the audio. Shortening the Input Chunk reduces this number.

#### res

This is the time it takes to convert data that is the sum of Input Chunk and Extra Data Length. Shortening both Input Chunk and Extra Data Length will reduce the number.

### Switch Model

### Switch Model

You can switch between uploaded models.
Information about the model is shown in [] under the name

1. Is the model considering f0(=pitch)?
   - f0: consider
   - nof0: don't consider
2. Sampling rate used to train the model
3. Number of feature channels used by the model
4. Clients used for learning
   - org: This is the model trained in [orginal-RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI).
   - webui: The model trained on [ddPn08-RVC](https://github.com/ddPn08/rvc-webui).

### Operation

A button is placed to perform operations on the model and server. and server.

#### export onnx

We can output an ONNX model. Converting a PyTorch model to an ONNX model can sometimes speed up inference.

#### download

Download the model. It is mainly used to get the results of model merging.

## Model Setting

#### Model Slot

You can choose which frame to set the model in. The set model can be switched with Switch Model in Server Control.

When setting up the model, you can choose to either load the file or download it from the internet. Depending on your choice, the available settings will change.

- file: Select a local file to load the model.
- from net: Download the model from the internet.

#### Model(.onnx or .pth)

If you set it to load from a file, it will be displayed.

Specify the trained model here. Required fields.
You can choose either ONNX format (.onnx) or PyTorch format (.pth).

- If trained with [orginal-RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI), it is in `/logs/weights`.
- If trained with [ddPn08-RVC](https://github.com/ddPn08/rvc-webui), it is in `/models/checkpoints`.

#### feature(.npy)

If you set it to load from a file, it will be displayed.

This is an additional function that brings the features extracted by HuBERT closer to the training data. Used in pairs with index(.index).

- If trained with [orginal-RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI), it is in `/logs/your-expetiment-name/total_fea.npy`.
- If trained with [ddPn08-RVC](https://github.com/ddPn08/rvc-webui), it is in `/models/checkpoints/your-model-name_index/your-model-name.0.big.npy`.

#### index(.index)

If you set it to load from a file, it will be displayed.

This is an additional function that brings the features extracted by HuBERT closer to the training data. Used in pairs with feature(.npy).

- If trained with [orginal-RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI), it is in `/logs/your-expetiment-name/total_fea.npy`.
- If trained with [ddPn08-RVC](https://github.com/ddPn08/rvc-webui), it is in `/models/checkpoints/your-model-name_index/your-model-name.0.big.npy`.

#### Select Model

If you choose to download from the internet, you will see the model to download. Please check the link to the terms of use before using it.

#### Default Tune

Enter the default value for how much the pitch of the voice should be converted. You can also convert during inference. Below is a guideline for the settings.

- +12 for male voice to female voice conversion
- -12 for female voice to male voice conversion

#### upload

After setting the above items, press to make the model ready for use.

#### select

When you set the option to download from the internet, the items above will be displayed. After setting the items above, press to activate the model.

## Speaker Setting

### Tuning

Adjust the pitch of your voice. Below is a guideline for the settings.

- +12 for male voice to female voice conversion
- -12 for female voice to male voice conversion

### index ratio

Specify the ratio to shift to the features used in training. Effective when both feature and index are set in Model Setting.
0 uses the output of HuBERT as it is, 1 brings it all back to the original features.
If the index ratio is greater than 0, the search may take a long time.

### Silent Threshold

The volume threshold for audio conversion. If the rms is smaller than this value, no voice conversion is performed and silence is returned.
(In this case, the conversion process is skipped, so the load is less.)

## Converter Setting

### InputChunk Num(128sample / chunk)

Decide how much length to cut and convert in one conversion. The higher the value, the more efficient the conversion, but the larger the buf value, the longer the maximum time before the conversion starts. The approximate time is displayed in buff:.

### Extra Data Length

Determines how much past audio to include in the input when converting audio. The longer the past voice is, the better the accuracy of the conversion, but the longer the res is, the longer the calculation takes.
(Probably because Transformer is a bottleneck, the calculation time will increase by the square of this length)

Detail is [here](https://github.com/w-okada/voice-changer/issues/154#issuecomment-1502534841)

### GPU

If you have 2 or more GPUs, you can choose your GPU here.

## Device Setting

Choose between client device mode and server device mode. You can only change it when the voice conversion is stopped.

For more details on each mode, please see [here](./tutorial_device_mode.md).

### Audio Input

Choose an input device

### Audio Output

Choose an output terminal

#### output record

It will only be displayed when in client device mode.

Audio is recorded from when you press start until you press stop.
Pressing this button does not start real-time conversion.
Press Server Control for real-time conversion

## Lab

You can do model merging.
Set the component amounts for each source model for the merge. Create a new model according to the ratio of the component amounts.

## Quality Control

### Noise Supression

On/Off of the browser's built-in noise removal function.

### Gain Control

- input: Increase or decrease the volume of the input audio to the model. 1 is the default value
- output: Increase or decrease the volume of the output audio from the model. 1 is the default value

### F0Detector

Choose an algorithm for extracting the pitch. You can choose from the following two types.

- Lightweight `pm`
- Highly accurate `harvest`

### Analyzer(Experimental)

Record input and output on the server side.
As for the input, the sound of the microphone is sent to the server and recorded as it is. It can be used to check the communication path from the microphone to the server.
For output, the data output from the model is recorded in the server. You can see how the model behaves (once you've verified that your input is correct).
