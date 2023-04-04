import React, { useMemo } from "react"
import { useAppState } from "../../001_provider/001_AppStateProvider"

export const NoiseControlRow = () => {
    const appState = useAppState()


    const noiseControlRow = useMemo(() => {
        return (
            <div className="body-row split-3-2-2-2-1 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Noise Suppression</div>
                <div>
                    <input type="checkbox" checked={appState.clientSetting.clientSetting.echoCancel} onChange={(e) => {
                        appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, echoCancel: e.target.checked })
                    }} /> echo cancel
                </div>
                <div>
                    <input type="checkbox" checked={appState.clientSetting.clientSetting.noiseSuppression} onChange={(e) => {
                        appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, noiseSuppression: e.target.checked })
                    }} /> suppression1
                </div>
                <div>
                    <input type="checkbox" checked={appState.clientSetting.clientSetting.noiseSuppression2} onChange={(e) => {
                        appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, noiseSuppression2: e.target.checked })
                    }} /> suppression2
                </div>
                <div className="body-button-container">
                </div>
            </div>
        )
    }, [
        appState.clientSetting.clientSetting.echoCancel,
        appState.clientSetting.clientSetting.noiseSuppression,
        appState.clientSetting.clientSetting.noiseSuppression2,
        appState.clientSetting.updateClientSetting
    ])

    return noiseControlRow
}