import { Framework } from "@dannadori/voice-changer-client-js"
import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"


export type ModelSwitchRowProps = {
}

export const ModelSwitchRow = (_props: ModelSwitchRowProps) => {
    const appState = useAppState()

    const modelSwitchRow = useMemo(() => {
        const slot = appState.serverSetting.serverSetting.modelSlotIndex

        const onSwitchModelClicked = async (index: number, filename: string) => {
            const framework: Framework = filename.endsWith(".onnx") ? "ONNX" : "PyTorch"
            console.log("Framework:::", filename, framework)

            // Quick hack for same slot is selected. 下３桁が実際のSlotID
            const dummyModelSlotIndex = (Math.floor(Date.now() / 1000)) * 1000 + index
            await appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, modelSlotIndex: dummyModelSlotIndex, framework: framework })
        }
        const modelOptions = appState.serverSetting.serverSetting.modelSlots.map((x, index) => {
            const className = index == slot ? "body-button-active left-margin-1" : "body-button left-margin-1"
            let filename = ""
            if (x.pyTorchModelFile && x.pyTorchModelFile.length > 0) {
                filename = x.pyTorchModelFile.replace(/^.*[\\\/]/, '')
                return <div key={index} className={className} onClick={() => { onSwitchModelClicked(index, filename) }}>{filename}</div>
            } else if (x.onnxModelFile && x.onnxModelFile.length > 0) {
                filename = x.onnxModelFile.replace(/^.*[\\\/]/, '')
                return <div key={index} className={className} onClick={() => { onSwitchModelClicked(index, filename) }}>{filename}</div>
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

