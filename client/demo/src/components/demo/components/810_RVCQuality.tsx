import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export type RVCQualityRowProps = {
}

export const RVCQualityRow = (_props: RVCQualityRowProps) => {
    const appState = useAppState()

    const trancateNumTresholdRow = useMemo(() => {
        const onRVCQualityChanged = (val: number) => {
            appState.serverSetting.updateServerSettings({
                ...appState.serverSetting.serverSetting,
                rvcQuality: val
            })
        }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">RCV Quality</div>
                <div className="body-input-container">
                    <select value={appState.serverSetting.serverSetting.rvcQuality} onChange={(e) => { onRVCQualityChanged(Number(e.target.value)) }}>
                        <option value="0" >low</option>
                        <option value="1" >high</option>
                    </select>
                </div>
            </div>
        )
    }, [appState.serverSetting.serverSetting, appState.serverSetting.updateServerSettings])

    return trancateNumTresholdRow
}