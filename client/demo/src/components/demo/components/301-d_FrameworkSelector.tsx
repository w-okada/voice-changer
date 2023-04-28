import { Framework } from "@dannadori/voice-changer-client-js"
import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"

export const FrameworkSelectorRow = () => {
    const appState = useAppState()
    const guiState = useGuiState()
    const frameworkSelectorRow = useMemo(() => {
        const slot = guiState.modelSlotNum
        const fileUploadSetting = appState.serverSetting.fileUploadSettings[slot]
        const currentValue = fileUploadSetting?.framework || Framework.PyTorch

        const onFrameworkChanged = (val: Framework) => {
            appState.serverSetting.setFileUploadSetting(slot, {
                ...appState.serverSetting.fileUploadSettings[slot],
                framework: val
            })
        }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-2">Framework</div>
                <div className="body-input-container">
                    <div className="body-select-container">
                        <select className="body-select" value={currentValue} onChange={(e) => {
                            onFrameworkChanged(e.target.value as Framework)
                        }}>
                            {
                                Object.values(Framework).map(x => {
                                    return <option key={x} value={x}>{x}</option>
                                })
                            }
                        </select>
                    </div>

                </div>
            </div>
        )
    }, [appState.serverSetting.fileUploadSettings, appState.serverSetting.setFileUploadSetting, guiState.modelSlotNum])

    return frameworkSelectorRow
}