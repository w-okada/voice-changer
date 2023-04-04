import React, { useMemo } from "react"
import { useAppRoot } from "../../001_provider/001_AppRootProvider"
import { useAppState } from "../../001_provider/001_AppStateProvider"

export const SilentThresholdRow = () => {
    const appState = useAppState()
    const { appGuiSettingState } = useAppRoot()

    const speakerSetting = appGuiSettingState.appGuiSetting.front.speakerSetting

    const silentThresholdRow = useMemo(() => {
        if (!speakerSetting.silentThresholdEnable) {
            return <></>
        }

        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Silent Threshold</div>
                <div>
                    <input type="range" className="body-item-input-slider" min="0.00000" max="0.001" step="0.00001" value={appState.serverSetting.serverSetting.silentThreshold} onChange={(e) => {
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