import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export type EnableEnhancerRowProps = {
}

export const EnableEnhancerRow = (_props: EnableEnhancerRowProps) => {
    const appState = useAppState()

    const clusterRatioRow = useMemo(() => {
        return (
            <>
                <div className="body-row split-3-7 left-padding-1 guided">
                    <div className="body-item-title left-padding-1 ">Enhancer</div>
                    <div className="body-input-container">
                        <select value={appState.serverSetting.serverSetting.enableEnhancer} onChange={(e) => {
                            appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, enableEnhancer: Number(e.target.value) })
                        }}>
                            <option value="0" >disable</option>
                            <option value="1" >enable</option>
                        </select>
                    </div>
                </div>
                <div className="body-row split-3-3-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-1 ">Enhancer Tune</div>
                    <div>
                        <input type="range" className="body-item-input-slider" min="0" max="10" step="1" value={appState.serverSetting.serverSetting.enhancerTune || 0} onChange={(e) => {
                            appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, enhancerTune: Number(e.target.value) })
                        }}></input>
                        <span className="body-item-input-slider-val">{appState.serverSetting.serverSetting.enhancerTune}</span>
                    </div>
                    <div className="body-button-container">
                    </div>
                </div>
            </>
        )
    }, [
        appState.serverSetting.serverSetting,
        appState.serverSetting.updateServerSettings
    ])

    return clusterRatioRow
}


