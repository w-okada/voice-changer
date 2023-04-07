import React, { useMemo } from "react"
import { Framework, OnnxExecutionProvider } from "@dannadori/voice-changer-client-js"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export type FrameworkRowProps = {
    showFramework: boolean,
}

export const FrameworkRow = (props: FrameworkRowProps) => {
    const appState = useAppState()

    const frameworkRow = useMemo(() => {
        if (!props.showFramework) {
            return <></>
        }

        const onFrameworkChanged = async (val: Framework) => {
            appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, framework: val })
        }

        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Framework</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.serverSetting.serverSetting.framework} onChange={(e) => {
                        onFrameworkChanged(e.target.value as
                            Framework)
                    }}>
                        {
                            Object.values(Framework).map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [appState.serverSetting.serverSetting.framework, appState.serverSetting.updateServerSettings])

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
    }, [appState.serverSetting.serverSetting.framework, appState.serverSetting.updateServerSettings])


    return (
        <>
            {frameworkRow}
            {onnxExecutorRow}
        </>

    )
}