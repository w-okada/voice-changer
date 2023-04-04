import React, { useMemo } from "react"
import { useAppRoot } from "../../001_provider/001_AppRootProvider"
import { useAppState } from "../../001_provider/001_AppStateProvider"

export const CrossFadeOffsetRateRow = () => {
    const appState = useAppState()
    const { appGuiSettingState } = useAppRoot()

    const advancedSetting = appGuiSettingState.appGuiSetting.front.advancedSetting

    const crossFadeOffsetRateRow = useMemo(() => {
        if (!advancedSetting.crossFadeOffsetRateEnable) {
            return <></>
        }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title  left-padding-1">Cross Fade Offset Rate</div>
                <div className="body-input-container">
                    <input type="number" min={0} max={1} step={0.1} value={appState.serverSetting.serverSetting.crossFadeOffsetRate} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, crossFadeOffsetRate: Number(e.target.value) })
                    }} />
                </div>
            </div>
        )
    }, [appState.serverSetting.serverSetting.crossFadeOffsetRate, appState.serverSetting.updateServerSettings])

    return crossFadeOffsetRateRow
}