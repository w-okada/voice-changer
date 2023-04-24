import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"

export const DefaultTuneRow = () => {
    const appState = useAppState()
    const guiState = useGuiState()
    const defaultTuneRow = useMemo(() => {
        const slot = guiState.modelSlotNum
        const fileUploadSetting = appState.serverSetting.fileUploadSettings[slot]
        if (!fileUploadSetting) {
            return <></>
        }
        const currentValue = fileUploadSetting.defaultTune

        const onDefaultTuneChanged = (val: number) => {
            appState.serverSetting.setFileUploadSetting(slot, {
                ...appState.serverSetting.fileUploadSettings[slot],
                defaultTune: val
            })
        }

        return (
            <div className="body-row split-3-2-1-4 left-padding-1 guided">
                <div className="body-item-title left-padding-2 ">Default Tune</div>
                <div>
                    <input type="range" className="body-item-input-slider" min="-50" max="50" step="1" value={currentValue} onChange={(e) => {
                        onDefaultTuneChanged(Number(e.target.value))
                    }}></input>
                    <span className="body-item-input-slider-val">{currentValue}</span>
                </div>
                <div>
                </div>
                <div className="body-button-container">
                </div>
            </div>
        )

    }, [appState.serverSetting.fileUploadSettings, guiState.modelSlotNum])

    return defaultTuneRow
}