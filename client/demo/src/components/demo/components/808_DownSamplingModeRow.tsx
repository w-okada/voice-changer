import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { DownSamplingMode } from "@dannadori/voice-changer-client-js"

export type DownSamplingModeRowProps = {
}

export const DownSamplingModeRow = (_props: DownSamplingModeRowProps) => {
    const appState = useAppState()

    const downSamplingModeRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">DownSamplingMode</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.workletNodeSetting.workletNodeSetting.downSamplingMode} onChange={(e) => {
                        appState.workletNodeSetting.updateWorkletNodeSetting({ ...appState.workletNodeSetting.workletNodeSetting, downSamplingMode: e.target.value as DownSamplingMode })
                    }}>
                        {
                            Object.values(DownSamplingMode).map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [appState.workletNodeSetting.workletNodeSetting, appState.workletNodeSetting.updateWorkletNodeSetting])

    return downSamplingModeRow
}