import React, { useMemo, useEffect } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export type F0FactorRowProps = {
}
export const F0FactorRow = (_props: F0FactorRowProps) => {
    const appState = useAppState()

    const f0FactorRow = useMemo(() => {

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
                    <input type="range" className="body-item-input-slider" min="0.1" max="5.0" step="0.01" value={appState.serverSetting.serverSetting.f0Factor || 0} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, f0Factor: Number(e.target.value) })
                    }}></input>
                    <span className="body-item-input-slider-val">{appState.serverSetting.serverSetting.f0Factor?.toFixed(2) || 0}</span>
                </div>
                <div className="body-item-text"></div>
                <div className="body-item-text">recommend: {recommendedF0Factor.toFixed(1)}</div>
            </div>
        )
    }, [appState.serverSetting.serverSetting.f0Factor, appState.serverSetting.serverSetting.srcId, appState.serverSetting.serverSetting.dstId, appState.clientSetting.clientSetting.correspondences, appState.serverSetting.updateServerSettings])


    useEffect(() => {
        const src = appState.clientSetting.clientSetting.correspondences?.find(x => {
            return x.sid == appState.serverSetting.serverSetting.srcId
        })
        const dst = appState.clientSetting.clientSetting.correspondences?.find(x => {
            return x.sid == appState.serverSetting.serverSetting.dstId
        })
        const recommendedF0Factor = dst && src ? dst.correspondence / src.correspondence : 0

        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, f0Factor: recommendedF0Factor })
    }, [appState.serverSetting.serverSetting.srcId, appState.serverSetting.serverSetting.dstId])

    return f0FactorRow
}

