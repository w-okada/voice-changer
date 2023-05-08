import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export type DiffSettingRowProps = {
}

export const DiffSettingRow = (_props: DiffSettingRowProps) => {
    const appState = useAppState()

    const diffSettingRow = useMemo(() => {


        return (
            <div className="body-row split-3-2-2-3 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Diff Setting</div>
                <div>
                    <span className="body-item-input-slider-label">Acc</span>
                    <input type="range" className="body-item-input-slider" min="1" max="20" step="1" value={appState.serverSetting.serverSetting.diffAcc} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, diffAcc: Number(e.target.value) })
                    }}></input>
                    <span className="body-item-input-slider-val">{appState.serverSetting.serverSetting.diffAcc}</span>
                </div>
                <div>
                    <span className="body-item-input-slider-label">kstep</span>
                    <input type="range" className="body-item-input-slider" min="21" max="300" step="1" value={appState.serverSetting.serverSetting.kStep} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, kStep: Number(e.target.value) })
                    }}></input>
                    <span className="body-item-input-slider-val">{appState.serverSetting.serverSetting.kStep}</span>
                </div>
                <div className="body-button-container">
                </div>
            </div>
        )
    }, [
        appState.serverSetting.serverSetting,
        appState.serverSetting.updateServerSettings
    ])

    return diffSettingRow
}


