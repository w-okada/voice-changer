import React, { useMemo } from "react"
import { fileSelector } from "@dannadori/voice-changer-client-js"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"

export const ConfigSelectRow = () => {
    const appState = useAppState()
    const guiState = useGuiState()

    const configSelectRow = useMemo(() => {
        const slot = guiState.modelSlotNum
        const configFilenameText = appState.serverSetting.fileUploadSettings[slot]?.configFile?.filename || appState.serverSetting.fileUploadSettings[slot]?.configFile?.file?.name || ""
        const onConfigFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".json") == false && file.name.endsWith(".yaml") == false) {
                alert("モデルファイルの拡張子はjsonである必要があります。")
                return
            }
            appState.serverSetting.setFileUploadSetting(slot, {
                ...appState.serverSetting.fileUploadSettings[slot],
                configFile: {
                    file: file
                }
            })
        }
        const onConfigFileClearClicked = () => {
            appState.serverSetting.setFileUploadSetting(slot, {
                ...appState.serverSetting.fileUploadSettings[slot],
                configFile: null
            })
        }

        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-2">Config(.json)</div>
                <div className="body-item-text">
                    <div>{configFilenameText}</div>
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onConfigFileLoadClicked}>select</div>
                    <div className="body-button left-margin-1" onClick={onConfigFileClearClicked}>clear</div>
                </div>
            </div>
        )
    }, [appState.serverSetting.fileUploadSettings, appState.serverSetting.setFileUploadSetting, guiState.modelSlotNum])

    return configSelectRow
}