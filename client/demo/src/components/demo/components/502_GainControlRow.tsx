import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export type GainControlRowProps = {
}

export const GainControlRow = (_props: GainControlRowProps) => {
    const appState = useAppState()


    const gainControlRow = useMemo(() => {
        return (
            <div className="body-row split-3-2-2-3 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Gain Control</div>
                <div>
                    <span className="body-item-input-slider-label">in</span>
                    <input type="range" className="body-item-input-slider" min="0.1" max="10.0" step="0.1" value={appState.clientSetting.clientSetting.inputGain} onChange={(e) => {
                        appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, inputGain: Number(e.target.value) })
                    }}></input>
                    <span className="body-item-input-slider-val">{appState.clientSetting.clientSetting.inputGain}</span>
                </div>
                <div>
                    <span className="body-item-input-slider-label">out</span>
                    <input type="range" className="body-item-input-slider" min="0.1" max="10.0" step="0.1" value={appState.clientSetting.clientSetting.outputGain} onChange={(e) => {
                        appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, outputGain: Number(e.target.value) })
                    }}></input>
                    <span className="body-item-input-slider-val">{appState.clientSetting.clientSetting.outputGain}</span>
                </div>
                <div className="body-button-container">
                </div>
            </div>
        )
    }, [
        appState.clientSetting.clientSetting.inputGain,
        appState.clientSetting.clientSetting.outputGain,
        appState.clientSetting.updateClientSetting
    ])

    return gainControlRow
}