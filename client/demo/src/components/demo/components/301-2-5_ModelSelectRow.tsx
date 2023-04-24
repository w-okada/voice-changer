import React, { useMemo } from "react"
import { fileSelector } from "@dannadori/voice-changer-client-js"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"


export const ModelSelectRow = () => {
    const appState = useAppState()
    const guiState = useGuiState()


    const onnxSelectRow = useMemo(() => {
        const slot = guiState.modelSlotNum
        const fileUploadSetting = appState.serverSetting.fileUploadSettings[slot]
        if (!fileUploadSetting) {
            return <></>
        }

        const onnxModelFilenameText = fileUploadSetting.onnxModel?.filename || fileUploadSetting.onnxModel?.file?.name || ""
        const pyTorchFilenameText = fileUploadSetting.pyTorchModel?.filename || fileUploadSetting.pyTorchModel?.file?.name || ""
        const modelFilenameText = onnxModelFilenameText + pyTorchFilenameText

        const onModelFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".onnx") == false && file.name.endsWith(".pth") == false) {
                alert("モデルファイルの拡張子は.onnxか.pthである必要があります。(Extension of the model file should be .onnx or .pth.)")
                return
            }
            if (file.name.endsWith(".onnx") == true) {
                appState.serverSetting.setFileUploadSetting(slot, {
                    ...appState.serverSetting.fileUploadSettings[slot],
                    onnxModel: {
                        file: file
                    },
                    pyTorchModel: null
                })
                return
            }
            if (file.name.endsWith(".pth") == true) {
                appState.serverSetting.setFileUploadSetting(slot, {
                    ...appState.serverSetting.fileUploadSettings[slot],
                    pyTorchModel: {
                        file: file
                    },
                    onnxModel: null
                })
                return
            }
        }
        const onModelFileClearClicked = () => {
            appState.serverSetting.setFileUploadSetting(slot, {
                ...appState.serverSetting.fileUploadSettings[slot],
                onnxModel: null,
                pyTorchModel: null
            })
        }

        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-2">Model(.onnx or .pth)</div>
                <div className="body-item-text">
                    <div>{modelFilenameText}</div>
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onModelFileLoadClicked}>select</div>
                    <div className="body-button left-margin-1" onClick={onModelFileClearClicked}>clear</div>
                </div>
            </div>
        )
    }, [appState.serverSetting.fileUploadSettings, appState.serverSetting.setFileUploadSetting, guiState.modelSlotNum])

    return onnxSelectRow
}