import React, { useMemo } from "react"
import { fileSelector } from "@dannadori/voice-changer-client-js"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"


export const IndexSelectRow = () => {
    const appState = useAppState()
    const guiState = useGuiState()


    const indexSelectRow = useMemo(() => {
        const slot = guiState.modelSlotNum
        const indexFilenameText = appState.serverSetting.fileUploadSettings[slot]?.index?.filename || appState.serverSetting.fileUploadSettings[slot]?.index?.file?.name || ""
        const onIndexFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".index") == false) {
                alert("Index file's extension should be .index")
                return
            }
            appState.serverSetting.setFileUploadSetting(slot, {
                ...appState.serverSetting.fileUploadSettings[slot],
                index: {
                    file: file
                }
            })
        }

        const onIndexFileClearClicked = () => {
            appState.serverSetting.setFileUploadSetting(slot, {
                ...appState.serverSetting.fileUploadSettings[slot],
                index: null
            })
        }

        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-2">index(.index)</div>
                <div className="body-item-text">
                    <div>{indexFilenameText}</div>
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onIndexFileLoadClicked}>select</div>
                    <div className="body-button left-margin-1" onClick={onIndexFileClearClicked}>clear</div>
                </div>
            </div>
        )
    }, [appState.serverSetting.fileUploadSettings, appState.serverSetting.setFileUploadSetting, guiState.modelSlotNum])

    return indexSelectRow
}