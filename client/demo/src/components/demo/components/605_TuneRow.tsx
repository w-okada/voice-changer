import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export type TuneRowProps = {
}
export const TuneRow = (_props: TuneRowProps) => {
    const appState = useAppState()

    const tuneRow = useMemo(() => {
        return (
            <div className="body-row split-3-2-2-3 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Tuning</div>
                <div>
                    <input type="range" className="body-item-input-slider" min="-50" max="50" step="1" value={appState.serverSetting.serverSetting.tran} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, tran: Number(e.target.value) })
                    }}></input>
                    <span className="body-item-input-slider-val">{appState.serverSetting.serverSetting.tran}</span>
                </div>
                <div>
                    <input type="checkbox" checked={appState.serverSetting.serverSetting.predictF0 == 1} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, predictF0: e.target.checked ? 1 : 0 })
                    }} /> predict f0
                </div>
                <div className="body-button-container">
                </div>
            </div>
        )
    }, [
        appState.serverSetting.serverSetting,
        appState.serverSetting.updateServerSettings
    ])

    return tuneRow
}