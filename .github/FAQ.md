# Frequently Asked Questions
Please read this FAQ before asking or making a bug report.

### 1. AMD GPU don't appear or not working
> Please download the **latest DirectML version**, use the **f0 det. rmvpe_onnx** and .ONNX models only! (.pth models do not work properly, use the "Export to ONNX" if you have a .pth model)

### 2. NVidia GPU don't appear or not working
> Make sure that the [NVidia CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) drivers are installed on your PC and up-to-date

### 3. High CPU usage
> Decrease your EXTRA value and put the index feature to 0

### 4. High Latency/Reponse
> Decrease your chunk value until you find a good mix of quality and response time

### 5. I'm hearing my voice without changes!
> Make sure to disable **passthru** mode
