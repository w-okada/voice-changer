import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export type NoiseScaleRowProps = {
}

export const NoiseScaleRow = (_props: NoiseScaleRowProps) => {
    const appState = useAppState()

    const noiseScaleRow = useMemo(() => {
        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Noise Scale</div>
                <div>
                    <input type="range" className="body-item-input-slider" min="0" max="1" step="0.1" value={appState.serverSetting.serverSetting.noiseScale || 0} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, noiseScale: Number(e.target.value) })
                    }}></input>
                    <span className="body-item-input-slider-val">{appState.serverSetting.serverSetting.noiseScale}</span>
                </div>
                <div className="body-button-container">
                </div>
            </div>
        )
    }, [
        appState.serverSetting.serverSetting,
        appState.serverSetting.updateServerSettings
    ])


    return noiseScaleRow
}