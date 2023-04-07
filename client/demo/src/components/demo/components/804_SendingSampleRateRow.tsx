import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { InputSampleRate } from "@dannadori/voice-changer-client-js"

export type SendingSampleRateRowProps = {
}

export const SendingSampleRateRow = (_props: SendingSampleRateRowProps) => {
    const appState = useAppState()

    const sendingSampleRateRow = useMemo(() => {

        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Sending Sample Rate</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.workletNodeSetting.workletNodeSetting.sendingSampleRate} onChange={(e) => {
                        appState.workletNodeSetting.updateWorkletNodeSetting({ ...appState.workletNodeSetting.workletNodeSetting, sendingSampleRate: Number(e.target.value) as InputSampleRate })
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, inputSampleRate: Number(e.target.value) as InputSampleRate })
                    }}>
                        {
                            Object.values(InputSampleRate).map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [appState.workletNodeSetting.workletNodeSetting, appState.workletNodeSetting.updateWorkletNodeSetting, appState.serverSetting.updateServerSettings])

    return sendingSampleRateRow
}