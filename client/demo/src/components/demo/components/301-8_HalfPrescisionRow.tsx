import React, { useMemo } from "react"
import { fileSelector } from "@dannadori/voice-changer-client-js"
import { useAppState } from "../../../001_provider/001_AppStateProvider"


export const HalfPrecisionRow = () => {
    const appState = useAppState()

    const halfPrecisionSelectRow = useMemo(() => {
        const onHalfPrecisionChanged = () => {
            appState.serverSetting.setFileUploadSetting({
                ...appState.serverSetting.fileUploadSetting,
                isHalf: !appState.serverSetting.fileUploadSetting.isHalf
            })
        }

        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-2">-</div>
                <div className="body-item-text">
                    <div></div>
                </div>
                <div className="body-button-container">
                    <input type="checkbox" checked={appState.serverSetting.fileUploadSetting.isHalf} onChange={() => onHalfPrecisionChanged()} /> half-precision
                </div>
            </div>
        )
    }, [appState.serverSetting.fileUploadSetting, appState.serverSetting.setFileUploadSetting])

    return halfPrecisionSelectRow
}