import React, { useMemo } from "react"
import { fileSelector } from "@dannadori/voice-changer-client-js"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"


export const FeatureSelectRow = () => {
    const appState = useAppState()
    const guiState = useGuiState()


    const featureSelectRow = useMemo(() => {
        const slot = guiState.modelSlotNum
        const featureFilenameText = appState.serverSetting.fileUploadSettings[slot]?.feature?.filename || appState.serverSetting.fileUploadSettings[slot]?.feature?.file?.name || ""
        const onFeatureFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".npy") == false) {
                alert("Feature file's extension should be npy")
                return
            }
            appState.serverSetting.setFileUploadSetting(slot, {
                ...appState.serverSetting.fileUploadSettings[slot],
                feature: {
                    file: file
                }
            })
        }

        const onFeatureFileClearClicked = () => {
            appState.serverSetting.setFileUploadSetting(slot, {
                ...appState.serverSetting.fileUploadSettings[slot],
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
    }, [appState.serverSetting.fileUploadSettings, appState.serverSetting.setFileUploadSetting, guiState.modelSlotNum])

    return featureSelectRow
}