import React, { useMemo } from "react"
import { fileSelector } from "@dannadori/voice-changer-client-js"
import { useAppState } from "../../../001_provider/001_AppStateProvider"


export const IndexSelectRow = () => {
    const appState = useAppState()

    const indexSelectRow = useMemo(() => {
        const indexFilenameText = appState.serverSetting.fileUploadSetting.index?.filename || appState.serverSetting.fileUploadSetting.index?.file?.name || ""
        const onIndexFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".index") == false) {
                alert("Index file's extension should be .index")
                return
            }
            appState.serverSetting.setFileUploadSetting({
                ...appState.serverSetting.fileUploadSetting,
                index: {
                    file: file
                }
            })
        }

        const onIndexFileClearClicked = () => {
            appState.serverSetting.setFileUploadSetting({
                ...appState.serverSetting.fileUploadSetting,
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
    }, [appState.serverSetting.fileUploadSetting, appState.serverSetting.setFileUploadSetting])

    return indexSelectRow
}