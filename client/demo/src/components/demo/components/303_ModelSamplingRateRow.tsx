import React, { useMemo } from "react"
import { fileSelector, ModelSamplingRate } from "@dannadori/voice-changer-client-js"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export type ModelSamplingRateRowProps = {
}

export const ModelSamplingRateRow = (_props: ModelSamplingRateRowProps) => {
    const appState = useAppState()

    const modelSamplingRateRow = useMemo(() => {
        const onModelSamplingRateChanged = (val: ModelSamplingRate) => {
            appState.serverSetting.updateServerSettings({
                ...appState.serverSetting.serverSetting,
                modelSamplingRate: val
            })
        }

        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-2">Model Sampling Rate</div>
                <div className="body-item-text">
                    <div></div>
                </div>
                <div className="body-button-container">
                    <select className="body-select" value={appState.serverSetting.serverSetting.modelSamplingRate} onChange={(e) => {
                        onModelSamplingRateChanged(e.target.value as unknown as ModelSamplingRate)
                    }}>
                        {
                            Object.values(ModelSamplingRate).map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>


                </div>
            </div>
        )
    }, [appState.serverSetting.serverSetting, appState.serverSetting.updateServerSettings])

    return modelSamplingRateRow
}