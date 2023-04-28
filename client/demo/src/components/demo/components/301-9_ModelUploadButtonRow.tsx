import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"

export const ModelUploadButtonRow = () => {
    const appState = useAppState()
    const guiState = useGuiState()
    const modelUploadButtonRow = useMemo(() => {
        const slot = guiState.modelSlotNum
        const onModelUploadClicked = async () => {
            appState.serverSetting.loadModel(slot)
        }

        const uploadButtonClassName = appState.serverSetting.isUploading ? "body-button-disabled" : "body-button"
        const uploadButtonAction = appState.serverSetting.isUploading ? () => { } : onModelUploadClicked
        const uploadButtonLabel = appState.serverSetting.isUploading ? "wait..." : "upload"
        const uploadingStatus = appState.serverSetting.isUploading ?
            appState.serverSetting.uploadProgress == 0 ? `loading model...(wait about 20sec)` : `uploading.... ${appState.serverSetting.uploadProgress.toFixed(1)}%` : ""


        const uploadedText = appState.serverSetting.fileUploadSettings[slot] == undefined ? "" : appState.serverSetting.fileUploadSettings[slot].uploaded ? "" : "not uploaded"
        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-2"></div>
                <div className="body-item-text">
                    {uploadingStatus}
                </div>
                <div className="body-button-container">
                    <div className={uploadButtonClassName} onClick={uploadButtonAction}>{uploadButtonLabel}</div>
                    <div className="body-item-text-em" >{uploadedText}</div>
                </div>
            </div>

        )
    }, [appState.serverSetting.isUploading, appState.serverSetting.uploadProgress, appState.serverSetting.loadModel, guiState.modelSlotNum, appState.serverSetting.fileUploadSettings])

    return modelUploadButtonRow
}