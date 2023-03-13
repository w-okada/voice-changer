import { OnnxExecutionProvider, Framework, fileSelector, Correspondence } from "@dannadori/voice-changer-client-js"
import React, { useState } from "react"
import { useMemo } from "react"
import { useAppState } from "./001_provider/001_AppStateProvider";
import { AnimationTypes, HeaderButton, HeaderButtonProps } from "./components/101_HeaderButton";

export type ServerSettingState = {
    modelSetting: JSX.Element;
}

export const useModelSettingArea = (): ServerSettingState => {
    const appState = useAppState()
    // const [showPyTorch, setShowPyTorch] = useState<boolean>(false)
    const [showPyTorch, setShowPyTorch] = useState<boolean>(true)

    const accodionButton = useMemo(() => {
        const accodionButtonProps: HeaderButtonProps = {
            stateControlCheckbox: appState.frontendManagerState.stateControls.openModelSettingCheckbox,
            tooltip: "Open/Close",
            onIcon: ["fas", "caret-up"],
            offIcon: ["fas", "caret-up"],
            animation: AnimationTypes.spinner,
            tooltipClass: "tooltip-right",
        };
        return <HeaderButton {...accodionButtonProps}></HeaderButton>;
    }, []);

    const uploadeModelRow = useMemo(() => {
        const onPyTorchFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".pth") == false) {
                alert("モデルファイルの拡張子はpthである必要があります。")
                return
            }
            appState.serverSetting.setFileUploadSetting({
                ...appState.serverSetting.fileUploadSetting,
                pyTorchModel: {
                    file: file
                }
            })
        }
        const onPyTorchFileClearClicked = () => {
            appState.serverSetting.setFileUploadSetting({
                ...appState.serverSetting.fileUploadSetting,
                pyTorchModel: null
            })
        }
        const onConfigFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".json") == false) {
                alert("モデルファイルの拡張子はjsonである必要があります。")
                return
            }
            appState.serverSetting.setFileUploadSetting({
                ...appState.serverSetting.fileUploadSetting,
                configFile: {
                    file: file
                }
            })
        }
        const onConfigFileClearClicked = () => {
            appState.serverSetting.setFileUploadSetting({
                ...appState.serverSetting.fileUploadSetting,
                configFile: null
            })
        }

        const onHubertFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".pth") == false) {
                alert("モデルファイルの拡張子はpthである必要があります。")
                return
            }
            appState.serverSetting.setFileUploadSetting({
                ...appState.serverSetting.fileUploadSetting,
                hubertTorchModel: {
                    file: file
                }
            })
        }

        const onHubertFileClearClicked = () => {
            appState.serverSetting.setFileUploadSetting({
                ...appState.serverSetting.fileUploadSetting,
                hubertTorchModel: null
            })
        }

        // const onOnnxFileLoadClicked = async () => {
        //     const file = await fileSelector("")
        //     if (file.name.endsWith(".onnx") == false) {
        //         alert("モデルファイルの拡張子はonnxである必要があります。")
        //         return
        //     }
        //     appState.serverSetting.setFileUploadSetting({
        //         ...appState.serverSetting.fileUploadSetting,
        //         onnxModel: {
        //             file: file
        //         }
        //     })
        // }
        // const onOnnxFileClearClicked = () => {
        //     appState.serverSetting.setFileUploadSetting({
        //         ...appState.serverSetting.fileUploadSetting,
        //         onnxModel: null
        //     })
        // }

        const onModelUploadClicked = async () => {
            appState.serverSetting.loadModel()
        }

        const uploadButtonClassName = appState.serverSetting.isUploading ? "body-button-disabled" : "body-button"
        const uploadButtonAction = appState.serverSetting.isUploading ? () => { } : onModelUploadClicked
        const uploadButtonLabel = appState.serverSetting.isUploading ? "wait..." : "upload"

        const configFilenameText = appState.serverSetting.fileUploadSetting.configFile?.filename || appState.serverSetting.fileUploadSetting.configFile?.file?.name || ""
        const hubertModelFilenameText = appState.serverSetting.fileUploadSetting.hubertTorchModel?.filename || appState.serverSetting.fileUploadSetting.hubertTorchModel?.file?.name || ""
        // const onnxModelFilenameText = appState.serverSetting.fileUploadSetting.onnxModel?.filename || appState.serverSetting.fileUploadSetting.onnxModel?.file?.name || ""

        const pyTorchFilenameText = appState.serverSetting.fileUploadSetting.pyTorchModel?.filename || appState.serverSetting.fileUploadSetting.pyTorchModel?.file?.name || ""

        const uploadingStatus = appState.serverSetting.isUploading ?
            appState.serverSetting.uploadProgress == 0 ? `loading model...(wait about 20sec)` : `uploading.... ${appState.serverSetting.uploadProgress}%` : ""

        return (
            <>
                <div className="body-row split-3-3-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-1">Model Uploader</div>
                    <div className="body-item-text">
                        <div></div>
                    </div>
                    <div className="body-item-text">
                        <div>
                            {/* <input type="checkbox" checked={showPyTorch} onChange={(e) => {
                                setShowPyTorch(e.target.checked)
                            }} /> enable PyTorch */}
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
                    <div className="body-item-title left-padding-2">hubert(.pth)</div>
                    <div className="body-item-text">
                        <div>{hubertModelFilenameText}</div>
                    </div>
                    <div className="body-button-container">
                        <div className="body-button" onClick={onHubertFileLoadClicked}>select</div>
                        <div className="body-button left-margin-1" onClick={onHubertFileClearClicked}>clear</div>
                    </div>
                </div>
                {/* <div className="body-row split-3-3-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-2">Onnx(.onnx)</div>
                    <div className="body-item-text">
                        <div>{onnxModelFilenameText}</div>
                    </div>
                    <div className="body-button-container">
                        <div className="body-button" onClick={onOnnxFileLoadClicked}>select</div>
                        <div className="body-button left-margin-1" onClick={onOnnxFileClearClicked}>clear</div>
                    </div>
                </div> */}
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
                        {uploadingStatus}
                    </div>
                    <div className="body-button-container">
                        <div className={uploadButtonClassName} onClick={uploadButtonAction}>{uploadButtonLabel}</div>
                    </div>
                </div>
            </>
        )
    }, [
        appState.serverSetting.fileUploadSetting,
        appState.serverSetting.loadModel,
        appState.serverSetting.isUploading,
        appState.serverSetting.uploadProgress,
        appState.clientSetting.clientSetting.correspondences,
        appState.serverSetting.updateServerSettings,
        appState.serverSetting.setFileUploadSetting,
        showPyTorch])

    const frameworkRow = useMemo(() => {
        return <></>
        // const onFrameworkChanged = async (val: Framework) => {
        //     appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, framework: val })
        // }
        // return (
        //     <div className="body-row split-3-7 left-padding-1 guided">
        //         <div className="body-item-title left-padding-1">Framework</div>
        //         <div className="body-select-container">
        //             <select className="body-select" value={appState.serverSetting.serverSetting.framework} onChange={(e) => {
        //                 onFrameworkChanged(e.target.value as
        //                     Framework)
        //             }}>
        //                 {
        //                     Object.values(Framework).map(x => {
        //                         return <option key={x} value={x}>{x}</option>
        //                     })
        //                 }
        //             </select>
        //         </div>
        //     </div>
        // )
    }, [appState.serverSetting.serverSetting.framework, appState.serverSetting.updateServerSettings])

    const onnxExecutionProviderRow = useMemo(() => {
        if (appState.serverSetting.serverSetting.framework != "ONNX") {
            return
        }
        const onOnnxExecutionProviderChanged = async (val: OnnxExecutionProvider) => {
            appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, onnxExecutionProvider: val })
        }
        console.log("setting", appState.serverSetting.serverSetting)
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
    }, [appState.serverSetting.serverSetting.framework, appState.serverSetting.serverSetting.onnxExecutionProvider, appState.serverSetting.updateServerSettings])

    const modelSetting = useMemo(() => {
        return (
            <>
                {appState.frontendManagerState.stateControls.openModelSettingCheckbox.trigger}
                <div className="partition">
                    <div className="partition-header">
                        <span className="caret">
                            {accodionButton}
                        </span>
                        <span className="title" onClick={() => { appState.frontendManagerState.stateControls.openModelSettingCheckbox.updateState(!appState.frontendManagerState.stateControls.openModelSettingCheckbox.checked()) }}>
                            Model Setting
                        </span>
                        <span></span>
                    </div>

                    <div className="partition-content">
                        {uploadeModelRow}
                        {frameworkRow}
                        {onnxExecutionProviderRow}
                    </div>
                </div>
            </>
        )
    }, [uploadeModelRow, frameworkRow, onnxExecutionProviderRow])


    return {
        modelSetting,
    }
}
