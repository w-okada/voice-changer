import React, { useMemo } from "react"
import { fileSelector } from "@dannadori/voice-changer-client-js"

import { useAppState } from "../../001_provider/001_AppStateProvider"
import { useGuiState } from "./001_GuiStateProvider"
import { useAppRoot } from "../../001_provider/001_AppRootProvider"

export const PyTorchSelectRow = () => {
    const appState = useAppState()
    const { appGuiSettingState } = useAppRoot()
    const modelSetting = appGuiSettingState.appGuiSetting.front.modelSetting
    const guiState = useGuiState()

    const pyTorchSelectRow = useMemo(() => {
        if (!modelSetting.pyTorchEnable) {
            return <></>
        }
        if (!guiState.showPyTorchModelUpload) {
            return <></>
        }

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

        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-2">PyTorch(.pth)</div>
                <div className="body-item-text">
                    <div>{pyTorchFilenameText}</div>
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onPyTorchFileLoadClicked}>select</div>
                    <div className="body-button left-margin-1" onClick={onPyTorchFileClearClicked}>clear</div>
                </div>
            </div>
        )
    }, [modelSetting.pyTorchEnable, guiState.showPyTorchModelUpload, appState.serverSetting.fileUploadSetting, appState.serverSetting.setFileUploadSetting])

    return pyTorchSelectRow
}