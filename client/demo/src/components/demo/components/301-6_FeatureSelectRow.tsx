import React, { useMemo } from "react"
import { fileSelector } from "@dannadori/voice-changer-client-js"
import { useAppState } from "../../../001_provider/001_AppStateProvider"


export const FeatureSelectRow = () => {
    const appState = useAppState()

    const featureSelectRow = useMemo(() => {
        const featureFilenameText = appState.serverSetting.fileUploadSetting.feature?.filename || appState.serverSetting.fileUploadSetting.feature?.file?.name || ""
        const onFeatureFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".npy") == false) {
                alert("Feature file's extension should be npy")
                return
            }
            appState.serverSetting.setFileUploadSetting({
                ...appState.serverSetting.fileUploadSetting,
                feature: {
                    file: file
                }
            })
        }

        const onFeatureFileClearClicked = () => {
            appState.serverSetting.setFileUploadSetting({
                ...appState.serverSetting.fileUploadSetting,
                feature: null
            })
        }

        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-2">feature(.npy)</div>
                <div className="body-item-text">
                    <div>{featureFilenameText}</div>
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onFeatureFileLoadClicked}>select</div>
                    <div className="body-button left-margin-1" onClick={onFeatureFileClearClicked}>clear</div>
                </div>
            </div>
        )
    }, [appState.serverSetting.fileUploadSetting, appState.serverSetting.setFileUploadSetting])

    return featureSelectRow
}