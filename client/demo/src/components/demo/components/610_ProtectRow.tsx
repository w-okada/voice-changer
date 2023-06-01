import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export type ProtectRowProps = {
}

export const ProtectRow = (_props: ProtectRowProps) => {
    const appState = useAppState()

    const clusterRatioRow = useMemo(() => {
        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">protect</div>
                <div>
                    <input type="range" className="body-item-input-slider" min="0" max="0.5" step="0.1" value={appState.serverSetting.serverSetting.protect || 0} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, protect: Number(e.target.value) })
                    }}></input>
                    <span className="body-item-input-slider-val">{appState.serverSetting.serverSetting.protect}</span>
                </div>
                <div className="body-button-container">
                </div>
            </div>
        )
    }, [
        appState.serverSetting.serverSetting,
        appState.serverSetting.updateServerSettings
    ])

    return clusterRatioRow
}