# Realtime Voice Changer Client for RVC Tutorial (v.1.5.3.7)

[Japanese/日本語](/tutorials/tutorial_rvc_ja_latest.md)

# Introduction

This application is client software for real-time voice conversion that supports various voice conversion models. This application support the models including RVC, MMVCv13, MMVCv15, So-vits-svcv40, etc. However, this document focus on [RVC(Retrieval-based-Voice-Conversion)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI) for voice conversion as the tutorial material. The basic operations for each model are essentially the same.

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

If you have the old version, be sure to unzip it into a separate folder.

### Mac version

It is launched as follows.

1. Unzip the downloaded file.

1. Next, run MMVCServerSIO by hold down the control key and clicking it (or right-click to run it). If a message appears stating that the developer cannot be verified, run it again by holding down the control key and clicking it (or right-click to run it). The terminal will open and the process will finish within a few seconds.

1. Next, execute the startHTTP.command by holding down the control key and clicking on it (or you can also right-click to run it). If a message appears stating that the developer cannot be verified, repeat the process by holding down the control key and clicking on it (or perform a right-click to run it). A terminal will open, and the launch process will begin.

- In other words, the key is to run both MMVCServerSIO and startHTTP.command. Moreover, you need to run MMVCServerSIO first.

If you have the old version, be sure to unzip it into a separate folder.

### Precautions when connecting remotely

When connecting remotely, please use `.bat` file (win) and `.command` file (mac) where http is replaced with https.

Access with Browser (currently only chrome is supported), then you can see gui.

### Console

When you run a .bat file (Windows) or .command file (Mac), a screen like the following will be displayed and various data will be downloaded from the Internet at the initial start-up. Depending on your environment, it may take 1-2 minutes in many cases.

![image](https://github.com/w-okada/voice-changer/assets/48346627/88a30097-2fb3-4c50-8bf1-19c41f27c481)

### GUI

Once the download of the required data is complete, a dialog like the one below will be displayed. If you wish, press the yellow icon to reward the developer with a cup of coffee. Pressing the Start button will make the dialog disappear.

![image](https://github.com/w-okada/voice-changer/assets/48346627/a8d12b5c-d1e8-4ca6-aed0-72cee6bb97c1)

# GUI Overview

Use this screen to operate.

![image](https://github.com/w-okada/voice-changer/assets/48346627/27add00d-5059-4cbf-a732-9deb6dc309ff)

# Quick start

You can immediately perform voice conversion using the data downloaded at startup.

## Operation

(1) To get started, click on the Model Selection area to select the model you would like to use. Once the model is loaded, the images of the characters will be displayed on the screen.

(2) Select the microphone (input) and speaker (output) you wish to use. If you are unfamiliar, we recommend selecting the client and then selecting your microphone and speaker. (We will explain the difference between server later).

(3) When you press the start button, the audio conversion will start after a few seconds of data loading. Try saying something into the microphone. You should be able to hear the converted audio from the speaker.

![image](https://github.com/w-okada/voice-changer/assets/48346627/883b296e-e5ca-4571-8fed-dcf7495ebb92)

## FAQ on Quick Start

Q1. The audio is becoming choppy and stuttering.

A1. It is possible that your PC's performance is not adequate. Try increasing the CHUNK value (as shown in Figure as A, for example, 1024). Also try setting F0 Det to dio (as shown in Figure as B).

![image](https://github.com/w-okada/voice-changer/assets/48346627/3c485d9b-53be-47c1-85d9-8663363b06f9)

Q2. The voice is not being converted.

A2. Refer to [this](https://github.com/w-okada/voice-changer/blob/master/tutorials/trouble_shoot_communication_ja.md) and identify where the problem lies, and consider a solution.

Q3. The pitch is off.

A3. Although it wasn't explained in the Quick Start, if the model is pitch-changeable, you can change it with TUNE. Please refer to the more detailed explanation below.

Q4. The window doesn't show up or the window shows up but the contents are not displayed. A console error such as `electron: Failed to load URL: http://localhost:18888/ with error: ERR_CONNECTION_REFUSED` is displayed.

A4. There is a possibility that the virus checker is running. Please wait or designate the folder to be excluded at your own risk.

Q5. `[4716:0429/213736.103:ERROR:gpu_init.cc(523)] Passthrough is not supported, GL is disabled, ANGLE is` is displayed

A5. This is an error produced by the library used by this application, but it does not have any effect, so please ignore it.

Q6. My AMD GPU isn't being used.

A6. Please use the DirectML version. Additionally, AMD GPUs are only enabled for ONNX models. You can judge this by the GPU utilization rate going up in the Performance Monitor.([see here](https://github.com/w-okada/voice-changer/issues/383))

Q7. onxxruntime is not launching and it's producing an error.

A7. It appears that an error occurs if the folder path contains unicode. Please extract to a path that does not use unicode (just alphanumeric characters). (Reference: https://github.com/w-okada/voice-changer/issues/528)

## Configurable items

## Title

![image](https://github.com/w-okada/voice-changer/assets/48346627/bb813fbb-4ea1-4c3b-87b0-da75b7eaac5e)

Icons are links.

| Icon                                                                                                                               | To                |
| :--------------------------------------------------------------------------------------------------------------------------------- | :---------------- |
| <img src="https://github.com/w-okada/rvc-trainer-docker/assets/48346627/97c18ca5-eee5-4be2-92a7-8092fff960f2" width="32"> Octocat  | github repository |
| <img src="https://github.com/w-okada/rvc-trainer-docker/assets/48346627/751164e4-7b7d-4d7e-b49c-1ad660bf7439" width="32"> question | manual            |
| <img src="https://github.com/w-okada/rvc-trainer-docker/assets/48346627/7bc188db-3aae-43eb-98a1-34aacc16173d" width="32"> spanner  | tools             |
| <img src="https://github.com/w-okada/rvc-trainer-docker/assets/48346627/5db16acc-e901-40d2-8fc2-1fb9fd67f59c" width="32"> coffee   | donation          |

### claer setting

Initialize configuration.

## Model Selection

![image](https://github.com/w-okada/voice-changer/assets/48346627/503eb581-a560-42b2-985b-d229d186eac8)

Select the model you wish to use.

By pressing the "edit" button, you can edit the list of models (model slots). Please refer to the model slots editing screen for more details.

## Main Control

![image](https://github.com/w-okada/voice-changer/assets/48346627/5a8dcf64-29d3-49cd-92f1-db7b539bfb3d)

A character image loaded on the left side will be displayed. The status of real-time voice changer is overlaid on the top left of the character image.

You can use the buttons and sliders on the right side to control various settings.

### status of real-time voice changer

The lag time from speaking to conversion is `buf + res` seconds. When adjusting, please adjust the buffer time to be longer than the res time.

#### vol

This is the volume after voice conversion.

#### buf

The length of each chunk in milliseconds when capturing audio. Shortening the CHUNK will decrease this number.

#### res

The time it takes to convert data with CHUNK and EXTRA added is measured. Decreasing either CHUNK or EXTRA will reduce the number.

### Control

#### start/stop button

Press "start" to begin voice conversion and "stop" to end it.

#### GAIN

- in: Change the volume of the inputted audio for the model.

- out: Change the volume of the converted audio.

#### TUNE

Enter a value for how much to convert the pitch of the voice. Conversion can also be done during inference. Below are some guidelines for settings.

- +12 for male voice to female voice conversion
- -12 for female voice to male voice conversion

#### INDEX (Only for RVC)

You can specify the rate of weight assigned to the features used in training. This is only valid for models which have an index file registered. 0 uses HuBERT's output as-is and 1 assigns all weights to the original features. If the index ratio is greater than 0, it may take longer to search.

#### Voice

Set the speaker of the audio conversion.

#### save setting

Save the settings specified. When the model is recalled again, the settings will be reflected. (Excluding some parts).

#### export to onnx

This output will convert the PyTorch model to ONNX. It is only valid if the loaded model is a RVC PyTorch model.

#### Others

The item that can be configured by the AI model used will vary. Please check the features and other information on the model manufacturer's website.

## Configuration

![image](https://github.com/w-okada/voice-changer/assets/48346627/cd04ba9f-f7e8-4a7e-8c93-cda3c81f3c1a)

You can review the action settings and transformation processes.

#### NOISE

You can switch the noise cancellation feature on and off, however it is only available in Client Device Mode.

- Echo: Echo Cancellation Function
- Sup1, Sup2: This is a noise suppression feature.

#### F0 Det (F0 Extractor)

Choose an algorithm for extracting the pitch. You can choose from the following options. AMD is available for only onnx.

| F0 Extractor | type  | description                 |
| ------------ | ----- | --------------------------- |
| dio          | cpu   | lightweight                 |
| harvest      | cpu   | High-precision              |
| crepe        | torch | GPU-enabled、high-precision |
| crepe full   | onnx  | GPU-enabled、high-precision |
| crepe tiny   | onnx  | GPU-enabled、lightweight    |
| rnvpe        | torch | GPU-enabled、high-precision |

#### S. Thresh (Noise Gate)

This is the threshold of the volume for performing speech conversion. When the rms is smaller than this value, speech conversion will be skipped and silence will be returned instead. (In this case, since the conversion process is skipped, the burden will not be so large.)

#### CHUNK (Input Chunk Num)

Decide how much length to cut and convert in one conversion. The higher the value, the more efficient the conversion, but the larger the buf value, the longer the maximum time before the conversion starts. The approximate time is displayed in buff:.

#### EXTRA (Extra Data Length)

Determines how much past audio to include in the input when converting audio. The longer the past voice is, the better the accuracy of the conversion, but the longer the res is, the longer the calculation takes.
(Probably because Transformer is a bottleneck, the calculation time will increase by the square of this length)

Detail is [here](https://github.com/w-okada/voice-changer/issues/154#issuecomment-1502534841)

#### GPU

You can select the GPU to use in the onnxgpu version.

In the onnxdirectML version, you can switch the GPU ON/OFF.

On DirectML Version, these buottns is displayed.

![image](https://github.com/w-okada/voice-changer/assets/48346627/5a66f237-e5b5-4819-9409-ff5eebb6e514)

- cpu: use cpu
- gpu0: use gpu0
- gpu1: use gpu1
- gpu2: use gpu2
- gpu3: use gpu3

Even if a GPU is not detected, gpu0 - gpu3 will still be displayed. If you specify a GPU that doesn't exist, the CPU will be used instead.[reference](https://github.com/w-okada/voice-changer/issues/410)

#### AUDIO

Choose the type of audio device you want to use. For more information, please refer to the [document](./tutorial_device_mode.md).

- Client: You can make use of the microphone input and speaker output with the GUI functions such as noise cancellation.
- Server: VCClient can directly control the microphone and speaker to minimize latency.

#### input

You can select a sound input device such as a microphone input. It's also possible to input from audio files (size limit applies).

For win user, system sound is available as input. Please note if you set the system sound as output, the sound loop occurs.

#### output

You can select audio output devices such as speakers and output.

#### monitor

In monitor mode, you can select audio output devices such as speaker output. This is only available in server device mode.

Please refer to [this document](tutorial_monitor_consept_ja.md) for an overview of the idea.

#### REC.

It will output the converted audio to a file.

### ServerIO Analizer

We can record and confirm the input audio to the speech conversion AI and the output audio from the speech conversion AI.

Please refer to [this document](trouble_shoot_communication_ja.md) for an overview of the idea.

#### SIO rec.

I will start/stop recording both the audio inputted into the voice conversion AI as well as the audio outputted from the voice conversion AI.

#### output

The AI will play back any audio that is input into it.

#### in

I will play the audio inputted to the speech conversion AI.

#### out

Play the audio output from the Speech Conversion AI.

### more...

You can do more advanced operations.

#### Merge Lab

It is possible to do synthesis of models.

#### Advanced Setting

You can set up more advanced settings.

#### Server Info

You can check the configuration of the current server.

# Model Slot Edit Screen

By pressing the edit button in the Model Slot Selection Area, you can edit the model slot.

![image](https://github.com/w-okada/voice-changer/assets/48346627/a4735a2e-540e-4e7c-aa70-ba5b91ff09eb)

## Icon Area

You can change the image by clicking on the icon.

## File Area

You can download the file by clicking on the file name.

## Upload Button

You can upload the model.

In the upload screen, you can select the voice changer type to upload.

You can go back to the Model Slot Edit Screen by pressing the back button.

![image](https://github.com/w-okada/voice-changer/assets/48346627/012c3585-0be2-4846-969a-882dcc07837b)

## Sample Button

You can download a sample.

You can go back to the Model Slot Edit Screen by pressing the back button.

![image](https://github.com/w-okada/voice-changer/assets/48346627/1c6e2529-af80-479a-8002-c37ebeb0c807)

## Edit Button

You can edit the details of the model slot.
