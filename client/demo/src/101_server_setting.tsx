import { OnnxExecutionProvider, Protocol, Framework, fileSelector } from "@dannadori/voice-changer-client-js"
import React from "react"
import { useMemo } from "react"
import { ClientState } from "./hooks/useClient"

export type UseServerSettingProps = {
    clientState: ClientState
}

export type ServerSettingState = {
    serverSetting: JSX.Element;
}

export const useServerSetting = (props: UseServerSettingProps): ServerSettingState => {
    const mmvcServerUrlRow = useMemo(() => {
        const onSetServerClicked = async () => {
            const input = document.getElementById("mmvc-server-url") as HTMLInputElement
            props.clientState.setSettingState({
                ...props.clientState.settingState,
                mmvcServerUrl: input.value
            })
        }
        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">MMVC Server</div>
                <div className="body-input-container">
                    <input type="text" defaultValue={props.clientState.settingState.mmvcServerUrl} id="mmvc-server-url" className="body-item-input" />
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onSetServerClicked}>set</div>
                </div>
            </div>
        )
    }, [props.clientState.settingState])

    const uploadeModelRow = useMemo(() => {
        const onPyTorchFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".pth") == false) {
                alert("モデルファイルの拡張子はpthである必要があります。")
                return
            }
            props.clientState.setSettingState({
                ...props.clientState.settingState,
                pyTorchModel: file
            })
        }
        const onPyTorchFileClearClicked = () => {
            props.clientState.setSettingState({
                ...props.clientState.settingState,
                pyTorchModel: null
            })
        }
        const onConfigFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".json") == false) {
                alert("モデルファイルの拡張子はjsonである必要があります。")
                return
            }
            props.clientState.setSettingState({
                ...props.clientState.settingState,
                configFile: file
            })
        }
        const onConfigFileClearClicked = () => {
            props.clientState.setSettingState({
                ...props.clientState.settingState,
                configFile: null
            })
        }
        const onOnnxFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".onnx") == false) {
                alert("モデルファイルの拡張子はonnxである必要があります。")
                return
            }
            props.clientState.setSettingState({
                ...props.clientState.settingState,
                onnxModel: file
            })
        }
        const onOnnxFileClearClicked = () => {
            props.clientState.setSettingState({
                ...props.clientState.settingState,
                onnxModel: null
            })
        }
        const onModelUploadClicked = async () => {
            props.clientState.loadModel()
        }

        return (
            <>
                <div className="body-row split-3-3-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-1">Model Uploader</div>
                    <div className="body-item-text">
                        <div></div>
                    </div>
                    <div className="body-item-text">
                        <div></div>
                    </div>
                </div>
                <div className="body-row split-3-3-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-2">PyTorch(.pth)</div>
                    <div className="body-item-text">
                        <div>{props.clientState.settingState.pyTorchModel?.name}</div>
                    </div>
                    <div className="body-button-container">
                        <div className="body-button" onClick={onPyTorchFileLoadClicked}>select</div>
                        <div className="body-button left-margin-1" onClick={onPyTorchFileClearClicked}>clear</div>
                    </div>
                </div>
                <div className="body-row split-3-3-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-2">Config(.json)</div>
                    <div className="body-item-text">
                        <div>{props.clientState.settingState.configFile?.name}</div>
                    </div>
                    <div className="body-button-container">
                        <div className="body-button" onClick={onConfigFileLoadClicked}>select</div>
                        <div className="body-button left-margin-1" onClick={onConfigFileClearClicked}>clear</div>
                    </div>
                </div>
                <div className="body-row split-3-3-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-2">Onnx(.onnx)</div>
                    <div className="body-item-text">
                        <div>{props.clientState.settingState.onnxModel?.name}</div>
                    </div>
                    <div className="body-button-container">
                        <div className="body-button" onClick={onOnnxFileLoadClicked}>select</div>
                        <div className="body-button left-margin-1" onClick={onOnnxFileClearClicked}>clear</div>
                    </div>
                </div>
                <div className="body-row split-3-3-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-2"></div>
                    <div className="body-item-text">
                        {props.clientState.isUploading ? `uploading.... ${props.clientState.uploadProgress}%` : ""}
                    </div>
                    <div className="body-button-container">
                        <div className="body-button" onClick={onModelUploadClicked}>upload</div>
                    </div>
                </div>
            </>
        )
    }, [
        props.clientState.settingState,
        props.clientState.loadModel,
        props.clientState.isUploading,
        props.clientState.uploadProgress])

    const protocolRow = useMemo(() => {
        const onProtocolChanged = async (val: Protocol) => {
            props.clientState.setSettingState({
                ...props.clientState.settingState,
                protocol: val
            })
        }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Protocol</div>
                <div className="body-select-container">
                    <select className="body-select" value={props.clientState.settingState.protocol} onChange={(e) => {
                        onProtocolChanged(e.target.value as
                            Protocol)
                    }}>
                        {
                            Object.values(Protocol).map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [props.clientState.settingState])

    const frameworkRow = useMemo(() => {
        const onFrameworkChanged = async (val: Framework) => {
            props.clientState.setSettingState({
                ...props.clientState.settingState,
                framework: val
            })
        }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Framework</div>
                <div className="body-select-container">
                    <select className="body-select" value={props.clientState.settingState.framework} onChange={(e) => {
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
    }, [props.clientState.settingState])

    const onnxExecutionProviderRow = useMemo(() => {
        if (props.clientState.settingState.framework != "ONNX") {
            return
        }
        const onOnnxExecutionProviderChanged = async (val: OnnxExecutionProvider) => {
            props.clientState.setSettingState({
                ...props.clientState.settingState,
                onnxExecutionProvider: val
            })
        }
        return (
            <div className="body-row split-3-7 left-padding-1">
                <div className="body-item-title left-padding-2">OnnxExecutionProvider</div>
                <div className="body-select-container">
                    <select className="body-select" value={props.clientState.settingState.onnxExecutionProvider} onChange={(e) => {
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
    }, [props.clientState.settingState])

    const serverSetting = useMemo(() => {
        return (
            <>
                <div className="body-row split-3-7 left-padding-1">
                    <div className="body-sub-section-title">Server Setting</div>
                    <div className="body-select-container">
                    </div>
                </div>
                {mmvcServerUrlRow}
                {uploadeModelRow}
                {frameworkRow}
                {onnxExecutionProviderRow}
                {protocolRow}
            </>
        )
    }, [mmvcServerUrlRow, uploadeModelRow, frameworkRow, onnxExecutionProviderRow, protocolRow])


    return {
        serverSetting,
    }
}
