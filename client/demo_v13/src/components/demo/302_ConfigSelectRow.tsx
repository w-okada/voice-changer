import React, { useMemo } from "react"
import { fileSelector } from "@dannadori/voice-changer-client-js"

import { useAppState } from "../../001_provider/001_AppStateProvider"
import { useAppRoot } from "../../001_provider/001_AppRootProvider"

export const ConfigSelectRow = () => {
    const appState = useAppState()
    const { appGuiSettingState } = useAppRoot()
    const modelSetting = appGuiSettingState.appGuiSetting.front.modelSetting

    const configSelectRow = useMemo(() => {
        if (!modelSetting.configRow) {
            return <></>
        }
        const configFilenameText = appState.serverSetting.fileUploadSetting.configFile?.filename || appState.serverSetting.fileUploadSetting.configFile?.file?.name || ""
        const onConfigFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".json") == false) {
                alert("モデルファイルの拡張子はjsonである必要があります。")
                return
            }
            appState.serverSetting.setFileUploadSetting({
                ...appState.serverSetting.fileUploadSetting,
                configFile: {
                    file: file
                }
            })
        }
        const onConfigFileClearClicked = () => {
            appState.serverSetting.setFileUploadSetting({
                ...appState.serverSetting.fileUploadSetting,
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
    }, [appState.serverSetting.fileUploadSetting, appState.serverSetting.setFileUploadSetting])

    return configSelectRow
}