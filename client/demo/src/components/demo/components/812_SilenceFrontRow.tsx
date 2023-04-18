import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export type SilenceFrontRowProps = {
}

export const SilenceFrontRow = (_props: SilenceFrontRowProps) => {
    const appState = useAppState()

    const trancateNumTresholdRow = useMemo(() => {
        const onSilenceFrontChanged = (val: number) => {
            appState.serverSetting.updateServerSettings({
                ...appState.serverSetting.serverSetting,
                silenceFront: val
            })
        }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Silence Front</div>
                <div className="body-input-container">
                    <select value={appState.serverSetting.serverSetting.silenceFront} onChange={(e) => { onSilenceFrontChanged(Number(e.target.value)) }}>
                        <option value="0" >off</option>
                        <option value="1" >on</option>
                    </select>
                </div>
            </div>
        )
    }, [appState.serverSetting.serverSetting, appState.serverSetting.updateServerSettings])

    return trancateNumTresholdRow
}