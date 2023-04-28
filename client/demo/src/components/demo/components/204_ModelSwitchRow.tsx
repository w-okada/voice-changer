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

            // Quick hack for same slot is selected. 下３桁が実際のSlotID
            const dummyModelSlotIndex = (Math.floor(Date.now() / 1000)) * 1000 + index
            await appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, modelSlotIndex: dummyModelSlotIndex, framework: framework })
        }
        const modelOptions = appState.serverSetting.serverSetting.modelSlots.map((x, index) => {
            const className = index == slot ? "body-button-active left-margin-1" : "body-button left-margin-1"
            let filename = ""
            if (x.pyTorchModelFile && x.pyTorchModelFile.length > 0) {
                filename = x.pyTorchModelFile.replace(/^.*[\\\/]/, '')
            } else if (x.onnxModelFile && x.onnxModelFile.length > 0) {
                filename = x.onnxModelFile.replace(/^.*[\\\/]/, '')
            } else {
                return <div key={index} ></div>
            }
            const f0str = x.f0 == true ? "f0" : "nof0"
            const srstr = Math.floor(x.samplingRate / 1000) + "K"
            const embedstr = x.embChannels
            const typestr = x.modelType == 0 ? "org" : "webui"
            const metadata = x.deprecated ? "[deprecated version]" : `[${f0str},${srstr},${embedstr},${typestr}]`


            return (
                <div key={index} className={className} onClick={() => { onSwitchModelClicked(index, filename) }}>
                    <div>
                        {filename}
                    </div>
                    <div>{metadata}</div>
                </div>
            )

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

