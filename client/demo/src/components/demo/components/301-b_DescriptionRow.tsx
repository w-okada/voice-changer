import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"

export const DescriptionRow = () => {
    const appState = useAppState()
    const guiState = useGuiState()
    const descriptionRow = useMemo(() => {
        const slot = guiState.modelSlotNum
        const fileUploadSetting = appState.serverSetting.fileUploadSettings[slot]
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-2">Model Desc.</div>
                <div className="body-input-container">
                    Tuning: {fileUploadSetting?.defaultTune || 0}
                </div>
            </div>
        )
    }, [appState.serverSetting.fileUploadSettings, guiState.modelSlotNum])

    return descriptionRow
}