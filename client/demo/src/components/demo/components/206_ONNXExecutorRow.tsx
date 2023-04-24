import { OnnxExecutionProvider, OnnxExporterInfo } from "@dannadori/voice-changer-client-js"
import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"


export type ONNXExecutorRowProps = {
}

export const ONNXExecutorRow = (_props: ONNXExecutorRowProps) => {
    const appState = useAppState()

    const onnxExecutorRow = useMemo(() => {
        if (appState.serverSetting.serverSetting.framework != "ONNX") {
            return <></>
        }
        const onOnnxExecutionProviderChanged = async (val: OnnxExecutionProvider) => {
            appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, onnxExecutionProvider: val })
        }

        return (
            <div className="body-row split-3-7 left-padding-1">
                <div className="body-item-title left-padding-2">OnnxExecutionProvider</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.serverSetting.serverSetting.onnxExecutionProvider} onChange={(e) => {
                        onOnnxExecutionProviderChanged(e.target.value as
                            OnnxExecutionProvider)
                    }}>
                        {
                            Object.values(OnnxExecutionProvider).map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
            </div>

        )
    }, [appState.getInfo, appState.serverSetting.serverSetting])

    return onnxExecutorRow
}

