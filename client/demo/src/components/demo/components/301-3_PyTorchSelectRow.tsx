import React, { useMemo } from "react"
import { fileSelector, OnnxExporterInfo } from "@dannadori/voice-changer-client-js"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"

export type PyTorchSelectRow = {
    showOnnxExportButton: boolean
}

export const PyTorchSelectRow = (props: PyTorchSelectRow) => {
    const appState = useAppState()
    const guiState = useGuiState()

    const pyTorchSelectRow = useMemo(() => {
        const pyTorchFilenameText = appState.serverSetting.fileUploadSetting.pyTorchModel?.filename || appState.serverSetting.fileUploadSetting.pyTorchModel?.file?.name || ""
        const onPyTorchFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".pth") == false) {
                alert("モデルファイルの拡張子はpthである必要があります。")
                return
            }
            appState.serverSetting.setFileUploadSetting({
                ...appState.serverSetting.fileUploadSetting,
                pyTorchModel: {
                    file: file
                }
            })
        }
        const onPyTorchFileClearClicked = () => {
            appState.serverSetting.setFileUploadSetting({
                ...appState.serverSetting.fileUploadSetting,
                pyTorchModel: null
            })
        }

        const onnxExportButtonAction = async () => {

            if (guiState.isConverting) {
                alert("cannot export onnx when voice conversion is enabled")
                return
            }
            document.getElementById("dialog")?.classList.add("dialog-container-show")
            guiState.stateControls.showWaitingCheckbox.updateState(true)
            const res = await appState.serverSetting.getOnnx() as OnnxExporterInfo
            const a = document.createElement("a")
            a.href = res.path
            a.download = res.filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            guiState.stateControls.showWaitingCheckbox.updateState(false)

        }

        const onnxExportButton = props.showOnnxExportButton ? <div className="body-button left-margin-1" onClick={onnxExportButtonAction}>export onnx</div> : <></>

        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-2">PyTorch(.pth)</div>
                <div className="body-item-text">
                    <div>{pyTorchFilenameText}</div>
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onPyTorchFileLoadClicked}>select</div>
                    <div className="body-button left-margin-1" onClick={onPyTorchFileClearClicked}>clear</div>
                    {onnxExportButton}
                </div>
            </div>
        )
    }, [guiState.showPyTorchModelUpload, appState.serverSetting.fileUploadSetting, appState.serverSetting.setFileUploadSetting, appState.serverSetting.serverSetting, appState.serverSetting.updateServerSettings, guiState.isConverting])

    return pyTorchSelectRow
}