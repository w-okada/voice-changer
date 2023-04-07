import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { Protocol } from "@dannadori/voice-changer-client-js"

export type ProtocolRowProps = {
}
export const ProtocolRow = (_props: ProtocolRowProps) => {
    const appState = useAppState()

    const protocolRow = useMemo(() => {
        const onProtocolChanged = async (val: Protocol) => {
            appState.workletNodeSetting.updateWorkletNodeSetting({ ...appState.workletNodeSetting.workletNodeSetting, protocol: val })
        }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Protocol</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.workletNodeSetting.workletNodeSetting.protocol} onChange={(e) => {
                        onProtocolChanged(e.target.value as
                            Protocol)
                    }}>
                        {
                            Object.values(Protocol).map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [appState.workletNodeSetting.workletNodeSetting, appState.workletNodeSetting.updateWorkletNodeSetting])

    return protocolRow
}