import React, { useMemo } from "react"
import { useAppState } from "../../001_provider/001_AppStateProvider"

export const F0FactorRow = () => {
    const appState = useAppState()
    const speakerSetting = appState.appGuiSettingState.appGuiSetting.front.speakerSetting

    const f0FactorRow = useMemo(() => {
        if (!speakerSetting.f0FactorEnable) {
            return <></>
        }

        const src = appState.clientSetting.clientSetting.correspondences?.find(x => {
            return x.sid == appState.serverSetting.serverSetting.srcId
        })
        const dst = appState.clientSetting.clientSetting.correspondences?.find(x => {
            return x.sid == appState.serverSetting.serverSetting.dstId
        })

        const recommendedF0Factor = dst && src ? dst.correspondence / src.correspondence : 0

        return (
            <div className="body-row split-3-2-1-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">F0 Factor</div>
                <div className="body-input-container">
                    <input type="range" className="body-item-input-slider" min="0.1" max="5.0" step="0.1" value={appState.serverSetting.serverSetting.f0Factor || 0} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, f0Factor: Number(e.target.value) })
                    }}></input>
                    <span className="body-item-input-slider-val">{appState.serverSetting.serverSetting.f0Factor?.toFixed(1) || 0}</span>
                </div>
                <div className="body-item-text"></div>
                <div className="body-item-text">recommend: {recommendedF0Factor.toFixed(1)}</div>
            </div>
        )
    }, [appState.serverSetting.serverSetting.f0Factor, appState.serverSetting.serverSetting.srcId, appState.serverSetting.serverSetting.dstId, appState.clientSetting.clientSetting.correspondences, appState.serverSetting.updateServerSettings])

    return f0FactorRow
}