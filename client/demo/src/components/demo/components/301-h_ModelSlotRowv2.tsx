import { MAX_MODEL_SLOT_NUM } from "@dannadori/voice-changer-client-js"
import React, { useMemo } from "react"
import { useGuiState } from "../001_GuiStateProvider"

export type ModelSlotRow2Props = {}
export const ModelSlotRow2 = (_prop: ModelSlotRow2Props) => {
    const guiState = useGuiState()
    const modelSlotRow = useMemo(() => {
        const slot = guiState.modelSlotNum
        const onModelSlotChanged = (val: number) => {
            guiState.setModelSlotNum(val)
        }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-2">Model Slot</div>
                <div className="body-input-container">
                    <select value={slot} onChange={(e) => { onModelSlotChanged(Number(e.target.value)) }}>
                        {Array(MAX_MODEL_SLOT_NUM).fill(0).map((_x, index) => {
                            return <option key={index} value={index} >{index}</option>
                        })}

                    </select>
                </div>
            </div>
        )
    }, [guiState.modelSlotNum])

    return modelSlotRow
}