import { OnnxExecutionProvider, Framework, fileSelector } from "@dannadori/voice-changer-client-js"
import React, { useState } from "react"
import { useMemo } from "react"
import { ClientState } from "@dannadori/voice-changer-client-js";

export type UseServerSettingProps = {
    clientState: ClientState
}

export type ServerSettingState = {
    serverSetting: JSX.Element;
}

export const useServerSettingArea = (props: UseServerSettingProps): ServerSettingState => {
    const [showPyTorch, setShowPyTorch] = useState<boolean>(true)
    const uploadeModelRow = useMemo(() => {
        const onPyTorchFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".pth") == false) {
                alert("モデルファイルの拡張子はpthである必要があります。")
                return
            }
            props.clientState.serverSetting.setFileUploadSetting({
                ...props.clientState.serverSetting.fileUploadSetting,
                pyTorchModel: {
                    file: file
                }
            })
        }
        const onPyTorchFileClearClicked = () => {
            props.clientState.serverSetting.setFileUploadSetting({
                ...props.clientState.serverSetting.fileUploadSetting,
                pyTorchModel: null
            })
        }
        const onConfigFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".json") == false) {
                alert("モデルファイルの拡張子はjsonである必要があります。")
                return
            }
            props.clientState.serverSetting.setFileUploadSetting({
                ...props.clientState.serverSetting.fileUploadSetting,
                configFile: {
                    file: file
                }
            })
        }
        const onConfigFileClearClicked = () => {
            props.clientState.serverSetting.setFileUploadSetting({
                ...props.clientState.serverSetting.fileUploadSetting,
                configFile: null
            })
        }
        const onOnnxFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".onnx") == false) {
                alert("モデルファイルの拡張子はonnxである必要があります。")
                return
            }
            props.clientState.serverSetting.setFileUploadSetting({
                ...props.clientState.serverSetting.fileUploadSetting,
                onnxModel: {
                    file: file
                }
            })
        }
        const onOnnxFileClearClicked = () => {
            props.clientState.serverSetting.setFileUploadSetting({
                ...props.clientState.serverSetting.fileUploadSetting,
                onnxModel: null
            })
        }
        const onModelUploadClicked = async () => {
            props.clientState.serverSetting.loadModel()
        }

        const uploadButtonClassName = props.clientState.serverSetting.isUploading ? "body-button-disabled" : "body-button"
        const uploadButtonAction = props.clientState.serverSetting.isUploading ? () => { } : onModelUploadClicked
        const uploadButtonLabel = props.clientState.serverSetting.isUploading ? "wait..." : "upload"

        const configFilenameText = props.clientState.serverSetting.fileUploadSetting.configFile?.filename || props.clientState.serverSetting.fileUploadSetting.configFile?.file?.name || ""
        const onnxModelFilenameText = props.clientState.serverSetting.fileUploadSetting.onnxModel?.filename || props.clientState.serverSetting.fileUploadSetting.onnxModel?.file?.name || ""
        const pyTorchFilenameText = props.clientState.serverSetting.fileUploadSetting.pyTorchModel?.filename || props.clientState.serverSetting.fileUploadSetting.pyTorchModel?.file?.name || ""

        return (
            <>
                <div className="body-row split-3-3-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-1">Model Uploader</div>
                    <div className="body-item-text">
                        <div></div>
                    </div>
                    <div className="body-item-text">
                        <div>
                            <input type="checkbox" checked={showPyTorch} onChange={(e) => {
                                setShowPyTorch(e.target.checked)
                            }} /> enable PyTorch
                        </div>
                    </div>
                </div>
                <div className="body-row split-3-3-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-2">Config(.json)</div>
                    <div className="body-item-text">
                        <div>{configFilenameText}</div>
                    </div>
                    <div className="body-button-container">
                        <div className="body-button" onClick={onConfigFileLoadClicked}>select</div>
                        <div className="body-button left-margin-1" onClick={onConfigFileClearClicked}>clear</div>
                    </div>
                </div>
                <div className="body-row split-3-3-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-2">Onnx(.onnx)</div>
                    <div className="body-item-text">
                        <div>{onnxModelFilenameText}</div>
                    </div>
                    <div className="body-button-container">
                        <div className="body-button" onClick={onOnnxFileLoadClicked}>select</div>
                        <div className="body-button left-margin-1" onClick={onOnnxFileClearClicked}>clear</div>
                    </div>
                </div>
                {showPyTorch ?
                    (
                        <div className="body-row split-3-3-4 left-padding-1 guided">
                            <div className="body-item-title left-padding-2">PyTorch(.pth)</div>
                            <div className="body-item-text">
                                <div>{pyTorchFilenameText}</div>
                            </div>
                            <div className="body-button-container">
                                <div className="body-button" onClick={onPyTorchFileLoadClicked}>select</div>
                                <div className="body-button left-margin-1" onClick={onPyTorchFileClearClicked}>clear</div>
                            </div>
                        </div>

                    )
                    :
                    (
                        <></>
                    )
                }
                <div className="body-row split-3-3-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-2"></div>
                    <div className="body-item-text">
                        {props.clientState.serverSetting.isUploading ? `uploading.... ${props.clientState.serverSetting.uploadProgress}%` : ""}
                    </div>
                    <div className="body-button-container">
                        <div className={uploadButtonClassName} onClick={uploadButtonAction}>{uploadButtonLabel}</div>
                    </div>
                </div>
            </>
        )
    }, [
        props.clientState.serverSetting.fileUploadSetting,
        props.clientState.serverSetting.loadModel,
        props.clientState.serverSetting.isUploading,
        props.clientState.serverSetting.uploadProgress,
        showPyTorch])

    const frameworkRow = useMemo(() => {
        const onFrameworkChanged = async (val: Framework) => {
            props.clientState.serverSetting.setFramework(val)
        }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Framework</div>
                <div className="body-select-container">
                    <select className="body-select" value={props.clientState.serverSetting.setting.framework} onChange={(e) => {
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
    }, [props.clientState.serverSetting.setting.framework, props.clientState.serverSetting.setFramework])

    const onnxExecutionProviderRow = useMemo(() => {
        if (props.clientState.serverSetting.setting.framework != "ONNX") {
            return
        }
        const onOnnxExecutionProviderChanged = async (val: OnnxExecutionProvider) => {
            props.clientState.serverSetting.setOnnxExecutionProvider(val)
        }
        return (
            <div className="body-row split-3-7 left-padding-1">
                <div className="body-item-title left-padding-2">OnnxExecutionProvider</div>
                <div className="body-select-container">
                    <select className="body-select" value={props.clientState.serverSetting.setting.onnxExecutionProvider} onChange={(e) => {
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
    }, [props.clientState.serverSetting.setting.framework, props.clientState.serverSetting.setting.onnxExecutionProvider, props.clientState.serverSetting.setOnnxExecutionProvider])

    const serverSetting = useMemo(() => {
        return (
            <>
                <div className="body-row split-3-7 left-padding-1">
                    <div className="body-sub-section-title">Server Setting</div>
                    <div className="body-select-container">
                    </div>
                </div>
                {uploadeModelRow}
                {frameworkRow}
                {onnxExecutionProviderRow}
            </>
        )
    }, [uploadeModelRow, frameworkRow, onnxExecutionProviderRow])


    return {
        serverSetting,
    }
}
