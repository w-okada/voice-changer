import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export type SilentThresholdRowProps = {
}

export const SilentThresholdRow = (_props: SilentThresholdRowProps) => {
    const appState = useAppState()

    const silentThresholdRow = useMemo(() => {
        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Silent Threshold</div>
                <div>
                    <input type="range" className="body-item-input-slider" min="0.00000" max="0.001" step="0.00001" value={appState.serverSetting.serverSetting.silentThreshold || 0} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, silentThreshold: Number(e.target.value) })
                    }}></input>
                    <span className="body-item-input-slider-val">{appState.serverSetting.serverSetting.silentThreshold}</span>
                </div>

                <div className="body-button-container">
                </div>
            </div>
        )
    }, [
        appState.serverSetting.serverSetting,
        appState.serverSetting.updateServerSettings
    ])


    return silentThresholdRow
}