import { DefaultVoiceChangerOptions, OnnxExecutionProvider, Protocol, Framework, fileSelector, ServerSettingKey } from "@dannadori/voice-changer-client-js"
import React from "react"
import { useMemo, useState } from "react"
import { ClientState } from "./hooks/useClient"

export type UseServerSettingProps = {
    clientState: ClientState
    loadModelFunc: (() => Promise<void>) | undefined
    uploadProgress: number,
    isUploading: boolean
}

export type ServerSettingState = {
    serverSetting: JSX.Element;
    mmvcServerUrl: string;
    pyTorchModel: File | null;
    configFile: File | null;
    onnxModel: File | null;
    framework: string;
    onnxExecutionProvider: OnnxExecutionProvider;
    protocol: Protocol;

}

export const useServerSetting = (props: UseServerSettingProps): ServerSettingState => {
    const [mmvcServerUrl, setMmvcServerUrl] = useState<string>(DefaultVoiceChangerOptions.mmvcServerUrl)
    const [pyTorchModel, setPyTorchModel] = useState<File | null>(null)
    const [configFile, setConfigFile] = useState<File | null>(null)
    const [onnxModel, setOnnxModel] = useState<File | null>(null)
    const [protocol, setProtocol] = useState<Protocol>("sio")
    const [onnxExecutionProvider, setOnnxExecutionProvider] = useState<OnnxExecutionProvider>("CPUExecutionProvider")
    const [framework, setFramework] = useState<Framework>("PyTorch")

    const mmvcServerUrlRow = useMemo(() => {
        const onSetServerClicked = async () => {
            const input = document.getElementById("mmvc-server-url") as HTMLInputElement
            setMmvcServerUrl(input.value)
        }
        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">MMVC Server</div>
                <div className="body-input-container">
                    <input type="text" defaultValue={mmvcServerUrl} id="mmvc-server-url" className="body-item-input" />
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onSetServerClicked}>set</div>
                </div>
            </div>
        )
    }, [])

    const uploadeModelRow = useMemo(() => {
        const onPyTorchFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".pth") == false) {
                alert("モデルファイルの拡張子はpthである必要があります。")
                return
            }
            setPyTorchModel(file)
        }
        const onPyTorchFileClearClicked = () => {
            setPyTorchModel(null)
        }
        const onConfigFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".json") == false) {
                alert("モデルファイルの拡張子はjsonである必要があります。")
                return
            }
            setConfigFile(file)
        }
        const onConfigFileClearClicked = () => {
            setConfigFile(null)
        }
        const onOnnxFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".onnx") == false) {
                alert("モデルファイルの拡張子はonnxである必要があります。")
                return
            }
            setOnnxModel(file)
        }
        const onOnnxFileClearClicked = () => {
            setOnnxModel(null)
        }
        const onModelUploadClicked = async () => {
            if (!props.loadModelFunc) {
                return
            }
            props.loadModelFunc()
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
                        <div>{pyTorchModel?.name}</div>
                    </div>
                    <div className="body-button-container">
                        <div className="body-button" onClick={onPyTorchFileLoadClicked}>select</div>
                        <div className="body-button left-margin-1" onClick={onPyTorchFileClearClicked}>clear</div>
                    </div>
                </div>
                <div className="body-row split-3-3-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-2">Config(.json)</div>
                    <div className="body-item-text">
                        <div>{configFile?.name}</div>
                    </div>
                    <div className="body-button-container">
                        <div className="body-button" onClick={onConfigFileLoadClicked}>select</div>
                        <div className="body-button left-margin-1" onClick={onConfigFileClearClicked}>clear</div>
                    </div>
                </div>
                <div className="body-row split-3-3-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-2">Onnx(.onnx)</div>
                    <div className="body-item-text">
                        <div>{onnxModel?.name}</div>
                    </div>
                    <div className="body-button-container">
                        <div className="body-button" onClick={onOnnxFileLoadClicked}>select</div>
                        <div className="body-button left-margin-1" onClick={onOnnxFileClearClicked}>clear</div>
                    </div>
                </div>
                <div className="body-row split-3-3-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-2"></div>
                    <div className="body-item-text">
                        {props.isUploading ? `uploading.... ${props.uploadProgress}%` : ""}
                    </div>
                    <div className="body-button-container">
                        <div className="body-button" onClick={onModelUploadClicked}>upload</div>
                    </div>
                </div>
            </>
        )
    }, [pyTorchModel, configFile, onnxModel, props.loadModelFunc, props.isUploading, props.uploadProgress])

    const protocolRow = useMemo(() => {
        const onProtocolChanged = async (val: Protocol) => {
            setProtocol(val)
        }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Protocol</div>
                <div className="body-select-container">
                    <select className="body-select" value={protocol} onChange={(e) => {
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
    }, [protocol])

    const frameworkRow = useMemo(() => {
        const onFrameworkChanged = async (val: Framework) => {
            setFramework(val)
        }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Framework</div>
                <div className="body-select-container">
                    <select className="body-select" value={framework} onChange={(e) => {
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
    }, [framework])

    const onnxExecutionProviderRow = useMemo(() => {
        if (framework != "ONNX") {
            return
        }
        const onOnnxExecutionProviderChanged = async (val: OnnxExecutionProvider) => {
            setOnnxExecutionProvider(val)
        }
        return (
            <div className="body-row split-3-7 left-padding-1">
                <div className="body-item-title left-padding-2">OnnxExecutionProvider</div>
                <div className="body-select-container">
                    <select className="body-select" value={onnxExecutionProvider} onChange={(e) => {
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
    }, [onnxExecutionProvider, framework, mmvcServerUrl])

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
        mmvcServerUrl,
        pyTorchModel,
        configFile,
        onnxModel,
        framework,
        onnxExecutionProvider,
        protocol,
    }

}
