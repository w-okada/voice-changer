import React, { useMemo, useState } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"

export type SampleDownloadControlRowProps = {}
export const SampleDownloadControlRow = (_props: SampleDownloadControlRowProps) => {
    const appState = useAppState()
    const guiState = useGuiState()

    const sampleDownloadControlRow = useMemo(() => {
        const slot = guiState.modelSlotNum
        const fileUploadSetting = appState.serverSetting.fileUploadSettings[slot]
        if (!fileUploadSetting) {
            return <></>
        }
        if (fileUploadSetting.isSampleMode == false) {
            return <></>
        }
        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Advanced Configuration</div>
                <div>
                    <input type="checkbox" checked={fileUploadSetting.rvcIndexDownload} onChange={(e) => {
                        appState.serverSetting.setFileUploadSetting(slot, { ...fileUploadSetting, rvcIndexDownload: e.target.checked })
                    }} /> useIndex
                </div>
                <div className="body-button-container">
                </div>
            </div>
        )


    }, [appState.serverSetting.fileUploadSettings, appState.serverSetting.setFileUploadSetting])

    return sampleDownloadControlRow
}