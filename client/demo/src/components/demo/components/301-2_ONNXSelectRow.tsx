import React, { useMemo } from "react"
import { fileSelector } from "@dannadori/voice-changer-client-js"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"

type ONNXSelectRowProps = {
    onlyWhenSelected: boolean
}

export const ONNXSelectRow = (props: ONNXSelectRowProps) => {
    const appState = useAppState()
    const guiState = useGuiState()


    const onnxSelectRow = useMemo(() => {
        const slot = guiState.modelSlotNum
        if (props.onlyWhenSelected && appState.serverSetting.fileUploadSettings[slot]?.framework != "ONNX") {
            return <></>
        }

        const onnxModelFilenameText = appState.serverSetting.fileUploadSettings[slot]?.onnxModel?.filename || appState.serverSetting.fileUploadSettings[slot]?.onnxModel?.file?.name || ""
        const onOnnxFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".onnx") == false) {
                alert("モデルファイルの拡張子はonnxである必要があります。")
                return
            }
            appState.serverSetting.setFileUploadSetting(slot, {
                ...appState.serverSetting.fileUploadSettings[slot],
                onnxModel: {
                    file: file
                }
            })
        }
        const onOnnxFileClearClicked = () => {
            appState.serverSetting.setFileUploadSetting(slot, {
                ...appState.serverSetting.fileUploadSettings[slot],
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
    }, [appState.serverSetting.fileUploadSettings, appState.serverSetting.setFileUploadSetting, guiState.modelSlotNum])

    return onnxSelectRow
}