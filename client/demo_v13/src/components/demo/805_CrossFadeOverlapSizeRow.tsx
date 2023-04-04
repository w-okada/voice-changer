import React, { useMemo } from "react"
import { useAppState } from "../../001_provider/001_AppStateProvider"
import { CrossFadeOverlapSize } from "@dannadori/voice-changer-client-js"

export const CrossFadeOverlapSizeRow = () => {
    const appState = useAppState()
    const advancedSetting = appState.appGuiSettingState.appGuiSetting.front.advancedSetting

    const crossFadeOverlapSizeRow = useMemo(() => {
        if (!advancedSetting.crossFadeOverlapSizeEnable) {
            return <></>
        }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title  left-padding-1">Cross Fade Overlap Size</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.serverSetting.serverSetting.crossFadeOverlapSize} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, crossFadeOverlapSize: Number(e.target.value) as CrossFadeOverlapSize })
                    }}>
                        {
                            Object.values(CrossFadeOverlapSize).map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [appState.serverSetting.serverSetting, appState.serverSetting.updateServerSettings])

    return crossFadeOverlapSizeRow
}