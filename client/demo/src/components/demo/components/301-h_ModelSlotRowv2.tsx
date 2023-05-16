import { MAX_MODEL_SLOT_NUM } from "@dannadori/voice-changer-client-js"
import React, { useMemo } from "react"
import { useGuiState } from "../001_GuiStateProvider"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export type ModelSlotRow2Props = {}
export const ModelSlotRow2 = (_prop: ModelSlotRow2Props) => {
    const guiState = useGuiState()
    const appState = useAppState()



    const modelSlotRow = useMemo(() => {
        const slot = guiState.modelSlotNum
        if (!appState.serverSetting.fileUploadSettings[slot]) {
            return <></>
        }

        const onModelSlotChanged = (val: number) => {
            guiState.setModelSlotNum(val)
        }
        const onModeChanged = (val: boolean) => {
            appState.serverSetting.fileUploadSettings[slot].isSampleMode = val
            appState.serverSetting.setFileUploadSetting(slot, {
                ...appState.serverSetting.fileUploadSettings[slot],
            })
        }

        const isSampleMode = appState.serverSetting.fileUploadSettings[slot].isSampleMode

        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">

                <div className="body-item-title left-padding-2">Model Slot</div>

                <div className="body-input-container">
                    <select value={slot} onChange={(e) => { onModelSlotChanged(Number(e.target.value)) }}>
                        {Array(MAX_MODEL_SLOT_NUM).fill(0).map((_x, index) => {
                            return <option key={index} value={index} >{index}</option>
                        })}
                    </select>
                    <div className="left-padding-1">
                        <input className="left-padding-1" type="radio" id="from-file" name="sample-mode" checked={isSampleMode == false} onChange={() => { onModeChanged(false) }} />
                        <label className="left-padding-05" htmlFor="from-file">file</label>
                    </div>
                    <div className="left-padding-1">
                        <input className="left-padding-1" type="radio" id="from-net" name="sample-mode" checked={isSampleMode == true} onChange={() => { onModeChanged(true) }} />
                        <label className="left-padding-05" htmlFor="from-net">from net</label>
                    </div>
                </div>
                <div></div>
            </div>
        )
    }, [guiState.modelSlotNum, appState.serverSetting.fileUploadSettings])

    return modelSlotRow
}