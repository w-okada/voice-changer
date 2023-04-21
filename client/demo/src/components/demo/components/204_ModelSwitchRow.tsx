import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"


export type ModelSwitchRowProps = {
}

export const ModelSwitchRow = (_props: ModelSwitchRowProps) => {
    const appState = useAppState()

    const modelSwitchRow = useMemo(() => {

        const onSwitchModelClicked = (index: number) => {
            const fileUploadSetting = appState.serverSetting.fileUploadSettings[index]

            appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, modelSlotIndex: index, tran: fileUploadSetting.defaultTune })

        }
        let filename = ""
        const modelOptions = appState.serverSetting.serverSetting.modelSlots.map((x, index) => {
            if (x.pyTorchModelFile && x.pyTorchModelFile.length > 0) {
                filename = x.pyTorchModelFile.replace(/^.*[\\\/]/, '')
                return <div key={index} className="body-button left-margin-1" onClick={() => { onSwitchModelClicked(index) }}>{filename}</div>
            } else if (x.onnxModelFile && x.onnxModelFile.length > 0) {
                filename = x.onnxModelFile.replace(/^.*[\\\/]/, '')
                return <div key={index} className="body-button left-margin-1" onClick={() => { onSwitchModelClicked(index) }}>{filename}</div>
            } else {
                return <div key={index} ></div>
            }

        })

        return (
            <>
                <div className="body-row split-3-7 left-padding-1 guided">
                    <div className="body-item-title left-padding-1">Swicth Model</div>
                    <div className="body-button-container">
                        {modelOptions}
                    </div>
                </div>
            </>
        )
    }, [appState.getInfo, appState.serverSetting.serverSetting])

    return modelSwitchRow
}

