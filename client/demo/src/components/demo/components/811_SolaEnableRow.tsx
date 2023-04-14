import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export type SolaEnableRowProps = {
}

export const SolaEnableRow = (_props: SolaEnableRowProps) => {
    const appState = useAppState()

    const trancateNumTresholdRow = useMemo(() => {
        const onSolaEnableChanged = (val: number) => {
            appState.serverSetting.updateServerSettings({
                ...appState.serverSetting.serverSetting,
                solaEnabled: val
            })
        }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Sola enable</div>
                <div className="body-input-container">
                    <select value={appState.serverSetting.serverSetting.solaEnabled} onChange={(e) => { onSolaEnableChanged(Number(e.target.value)) }}>
                        <option value="0" >disable</option>
                        <option value="1" >enable</option>
                    </select>
                </div>
            </div>
        )
    }, [appState.serverSetting.serverSetting, appState.serverSetting.updateServerSettings])

    return trancateNumTresholdRow
}