import React, { useMemo } from "react"
import { fileSelector } from "@dannadori/voice-changer-client-js"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export const ONNXSelectRow = () => {
    const appState = useAppState()

    const onnxSelectRow = useMemo(() => {
        const onnxModelFilenameText = appState.serverSetting.fileUploadSetting.onnxModel?.filename || appState.serverSetting.fileUploadSetting.onnxModel?.file?.name || ""
        const onOnnxFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".onnx") == false) {
                alert("モデルファイルの拡張子はonnxである必要があります。")
                return
            }
            appState.serverSetting.setFileUploadSetting({
                ...appState.serverSetting.fileUploadSetting,
                onnxModel: {
                    file: file
                }
            })
        }
        const onOnnxFileClearClicked = () => {
            appState.serverSetting.setFileUploadSetting({
                ...appState.serverSetting.fileUploadSetting,
                onnxModel: null
            })
        }

        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-2">Onnx(.onnx)</div>
                <div className="body-item-text">
                    <div>{onnxModelFilenameText}</div>
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onOnnxFileLoadClicked}>select</div>
                    <div className="body-button left-margin-1" onClick={onOnnxFileClearClicked}>clear</div>
                </div>
            </div>
        )
    }, [appState.serverSetting.fileUploadSetting, appState.serverSetting.setFileUploadSetting])

    return onnxSelectRow
}