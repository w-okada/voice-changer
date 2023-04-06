import React, { useMemo } from "react"
import { useAppRoot } from "../../001_provider/001_AppRootProvider"

import { useAppState } from "../../001_provider/001_AppStateProvider"

export const ModelUploadButtonRow = () => {
    const appState = useAppState()
    const { appGuiSettingState } = useAppRoot()

    const modelSetting = appGuiSettingState.appGuiSetting.front.modelSetting

    const modelUploadButtonRow = useMemo(() => {
        if (!modelSetting.uploadRow) {
            return <></>
        }
        const onModelUploadClicked = async () => {
            appState.serverSetting.loadModel()
        }

        const uploadButtonClassName = appState.serverSetting.isUploading ? "body-button-disabled" : "body-button"
        const uploadButtonAction = appState.serverSetting.isUploading ? () => { } : onModelUploadClicked
        const uploadButtonLabel = appState.serverSetting.isUploading ? "wait..." : "upload"
        const uploadingStatus = appState.serverSetting.isUploading ?
            appState.serverSetting.uploadProgress == 0 ? `loading model...(wait about 20sec)` : `uploading.... ${appState.serverSetting.uploadProgress}%` : ""


        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-2"></div>
                <div className="body-item-text">
                    {uploadingStatus}
                </div>
                <div className="body-button-container">
                    <div className={uploadButtonClassName} onClick={uploadButtonAction}>{uploadButtonLabel}</div>
                </div>
            </div>

        )
    }, [appState.serverSetting.isUploading, appState.serverSetting.uploadProgress, appState.serverSetting.loadModel])

    return modelUploadButtonRow
}