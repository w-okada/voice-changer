# Frequently Asked Questions
Please read this FAQ before asking or making a bug report.

### General fixes:
*Do all these steps so you can fix some issues.*
- Restart the application
- Go to your Windows %AppData% (Win + R, then put %appdata% and press Enter) and delete the "**voice-changer-native-client**" folder
- Extract your .zip to a new location and avoid folders with space or specials characters (also avoid long file paths)
- If you don't have a GPU or have a too old GPU, try using the [Colab Version](https://colab.research.google.com/github/w-okada/voice-changer/blob/master/Realtime_Voice_Changer_on_Colab.ipynb) instead

### 1. AMD GPU don't appear or not working
> Please download the **latest DirectML version**, use the **f0 det. rmvpe_onnx** and .ONNX models only! (.pth models do not work properly, use the "Export to ONNX" it can take a while)

### 2. NVidia GPU don't appear or not working
> Make sure that the [NVidia CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) drivers are installed on your PC and up-to-date

### 3. High CPU usage
> Decrease your EXTRA value and put the index feature to 0

### 4. High Latency
> Decrease your chunk value until you find a good mix of quality and response time

### 5. I'm hearing my voice without changes
> Make sure to disable **passthru** mode
