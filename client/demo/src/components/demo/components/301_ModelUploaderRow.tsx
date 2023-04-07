import React, { useMemo, useEffect } from "react"
import { useGuiState } from "../001_GuiStateProvider"
import { ConfigSelectRow } from "./301-1_ConfigSelectRow"
import { ONNXSelectRow } from "./301-2_ONNXSelectRow"
import { PyTorchSelectRow } from "./301-3_PyTorchSelectRow"
import { CorrespondenceSelectRow } from "./301-4_CorrespondenceSelectRow"
import { PyTorchClusterSelectRow } from "./301-5_PyTorchClusterSelectRow"
import { ModelUploadButtonRow } from "./301-9_ModelUploadButtonRow"

export type ModelUploaderRowProps = {
    showConfig: boolean
    showOnnx: boolean
    showPyTorch: boolean
    showCorrespondence: boolean
    showPyTorchCluster: boolean

    defaultEnablePyTorch: boolean
}

export const ModelUploaderRow = (props: ModelUploaderRowProps) => {
    const guiState = useGuiState()
    useEffect(() => {
        guiState.setShowPyTorchModelUpload(props.defaultEnablePyTorch)
    }, [])

    const modelUploaderRow = useMemo(() => {
        return (
            <>
                <div className="body-row split-3-3-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-1">Model Uploader</div>
                    <div className="body-item-text">
                        <div></div>
                    </div>
                    <div className="body-item-text">
                        <div>
                            <input type="checkbox" checked={guiState.showPyTorchModelUpload} onChange={(e) => {
                                guiState.setShowPyTorchModelUpload(e.target.checked)
                            }} /> enable PyTorch
                        </div>
                    </div>
                </div>
                {props.showConfig ? <ConfigSelectRow /> : <></>}
                {props.showOnnx ? <ONNXSelectRow /> : <></>}
                {props.showPyTorch && guiState.showPyTorchModelUpload ? <PyTorchSelectRow /> : <></>}
                {props.showCorrespondence ? <CorrespondenceSelectRow /> : <></>}
                {props.showPyTorchCluster ? <PyTorchClusterSelectRow /> : <></>}
                <ModelUploadButtonRow />
            </>
        )
    }, [guiState.showPyTorchModelUpload])

    return modelUploaderRow
}