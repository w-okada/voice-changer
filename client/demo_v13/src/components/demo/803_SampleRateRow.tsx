import React, { useMemo } from "react"
import { useAppState } from "../../001_provider/001_AppStateProvider"
import { SampleRate } from "@dannadori/voice-changer-client-js"

export const SampleRateRow = () => {
    const appState = useAppState()
    const advancedSetting = appState.appGuiSettingState.appGuiSetting.front.advancedSetting

    const sampleRateRow = useMemo(() => {
        if (!advancedSetting.sampleRateEnable) {
            return <></>
        }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Sample Rate</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.clientSetting.clientSetting.sampleRate} onChange={(e) => {
                        appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, sampleRate: Number(e.target.value) as SampleRate })
                    }}>
                        {
                            Object.values(SampleRate).map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [appState.clientSetting.clientSetting, appState.clientSetting.updateClientSetting])
    return sampleRateRow
}