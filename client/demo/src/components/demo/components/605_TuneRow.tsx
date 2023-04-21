import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"

export type TuneRowProps = {
    showPredictF0: boolean
    showSetDefault: boolean
}
export const TuneRow = (props: TuneRowProps) => {
    const appState = useAppState()
    const guiState = useGuiState()
    const tuneRow = useMemo(() => {
        const slot = guiState.modelSlotNum
        const predictF0 = props.showPredictF0 ?
            <>
                <input type="checkbox" checked={appState.serverSetting.serverSetting.predictF0 == 1} onChange={(e) => {
                    appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, predictF0: e.target.checked ? 1 : 0 })
                }} /> predict f0
            </> :
            <></>

        const showSetDefault = props.showSetDefault ?
            <>
                <div className="body-button" onClick={() => {
                    appState.serverSetting.updateDefaultTune(slot, appState.serverSetting.serverSetting.tran)
                }}>set model default</div>
            </> :
            <></>

        return (
            <div className="body-row split-3-2-1-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Tuning</div>
                <div>
                    <input type="range" className="body-item-input-slider" min="-50" max="50" step="1" value={appState.serverSetting.serverSetting.tran || 0} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, tran: Number(e.target.value) })
                    }}></input>
                    <span className="body-item-input-slider-val">{appState.serverSetting.serverSetting.tran}</span>
                </div>
                <div>
                    {predictF0}
                </div>
                <div className="body-button-container">
                    {showSetDefault}
                </div>
            </div>
        )
    }, [
        appState.serverSetting.serverSetting,
        appState.serverSetting.updateServerSettings,
        guiState.modelSlotNum

    ])

    return tuneRow
}