import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"


export const HalfPrecisionRow = () => {
    const appState = useAppState()
    const guiState = useGuiState()

    const halfPrecisionSelectRow = useMemo(() => {
        const slot = guiState.modelSlotNum
        const fileUploadSetting = appState.serverSetting.fileUploadSettings[slot]
        if (!fileUploadSetting) {
            return <></>
        }
        const currentValue = fileUploadSetting ? fileUploadSetting.isHalf : true
        const onHalfPrecisionChanged = () => {
            appState.serverSetting.setFileUploadSetting(slot, {
                ...appState.serverSetting.fileUploadSettings[slot],
                isHalf: !appState.serverSetting.fileUploadSettings[slot].isHalf
            })
        }

        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-2">-</div>
                <div className="body-item-text">
                    <input type="checkbox" checked={currentValue} onChange={() => onHalfPrecisionChanged()} /> half-precision
                </div>
                <div className="body-button-container">
                </div>
            </div>
        )
    }, [appState.serverSetting.fileUploadSettings, appState.serverSetting.setFileUploadSetting, guiState.modelSlotNum])

    return halfPrecisionSelectRow
}