import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { DiffMethod } from "@dannadori/voice-changer-client-js"

export type DiffMethodRowProps = {
}

export const DiffMethodRow = (_props: DiffMethodRowProps) => {
    const appState = useAppState()

    const diffMethodRow = useMemo(() => {
        const onDiffMethodChanged = (val: DiffMethod) => {
            appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, diffMethod: val })
        }

        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Diff Method</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.serverSetting.serverSetting.diffMethod} onChange={(e) => {
                        onDiffMethodChanged(e.target.value as DiffMethod)
                    }}>
                        {
                            Object.values(DiffMethod).map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
            </div>

        )
    }, [
        appState.serverSetting.serverSetting,
        appState.serverSetting.updateServerSettings
    ])

    return diffMethodRow
}


