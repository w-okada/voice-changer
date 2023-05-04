import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export type EnableDirectMLRowProps = {
}

export const EnableDirectMLRow = (_props: EnableDirectMLRowProps) => {
    const appState = useAppState()

    const enableDirctMLRow = useMemo(() => {
        const onEnableDirectMLChanged = (val: number) => {
            appState.serverSetting.updateServerSettings({
                ...appState.serverSetting.serverSetting,
                enableDirectML: val
            })
        }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">DirectML(experimental)</div>
                <div className="body-input-container">
                    <select value={appState.serverSetting.serverSetting.enableDirectML} onChange={(e) => { onEnableDirectMLChanged(Number(e.target.value)) }}>
                        <option value="0" >off</option>
                        <option value="1" >on</option>
                    </select>
                </div>
            </div>
        )
    }, [appState.serverSetting.serverSetting, appState.serverSetting.updateServerSettings])

    return enableDirctMLRow
}