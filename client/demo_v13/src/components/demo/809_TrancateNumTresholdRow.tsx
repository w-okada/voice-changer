import React, { useMemo } from "react"
import { useAppRoot } from "../../001_provider/001_AppRootProvider"
import { useAppState } from "../../001_provider/001_AppStateProvider"

export const TrancateNumTresholdRow = () => {
    const appState = useAppState()
    const { appGuiSettingState } = useAppRoot()

    const advancedSetting = appGuiSettingState.appGuiSetting.front.advancedSetting

    const trancateNumTresholdRow = useMemo(() => {
        if (!advancedSetting.trancateNumTresholdEnable) {
            return <></>
        }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Trancate Num</div>
                <div className="body-input-container">
                    <input type="number" min={5} max={300} step={1} value={appState.workletSetting.setting.numTrancateTreshold} onChange={(e) => {
                        appState.workletSetting.setSetting({
                            ...appState.workletSetting.setting,
                            numTrancateTreshold: Number(e.target.value)
                        })
                    }} />
                </div>
            </div>
        )
    }, [appState.workletNodeSetting.workletNodeSetting, appState.workletNodeSetting.updateWorkletNodeSetting])

    return trancateNumTresholdRow
}