import React, { useMemo, useEffect } from "react"
import { useGuiState } from "../001_GuiStateProvider"
import { ConfigSelectRow } from "./301-1_ConfigSelectRow"
import { ONNXSelectRow } from "./301-2_ONNXSelectRow"
import { PyTorchSelectRow } from "./301-3_PyTorchSelectRow"
import { CorrespondenceSelectRow } from "./301-4_CorrespondenceSelectRow"
import { PyTorchClusterSelectRow } from "./301-5_PyTorchClusterSelectRow"
import { FeatureSelectRow } from "./301-6_FeatureSelectRow"
import { IndexSelectRow } from "./301-7_IndexSelectRow"
import { HalfPrecisionRow } from "./301-8_HalfPrescisionRow"
import { ModelUploadButtonRow } from "./301-9_ModelUploadButtonRow"
import { ModelSlotRow } from "./301-a_ModelSlotRow"
import { DefaultTuneRow } from "./301-c_DefaultTuneRow"

export type ModelUploaderRowProps = {
    showModelSlot: boolean
    showConfig: boolean
    showOnnx: boolean
    showPyTorch: boolean
    showCorrespondence: boolean
    showPyTorchCluster: boolean

    showFeature: boolean
    showIndex: boolean
    showHalfPrecision: boolean
    showDescription: boolean
    showDefaultTune: boolean

    defaultEnablePyTorch: boolean

    showOnnxExportButton: boolean
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
                <ModelSlotRow />
                {props.showConfig ? <ConfigSelectRow /> : <></>}
                {props.showOnnx ? <ONNXSelectRow /> : <></>}
                {props.showPyTorch && guiState.showPyTorchModelUpload ? <PyTorchSelectRow showOnnxExportButton={props.showOnnxExportButton || false} /> : <></>}
                {props.showCorrespondence ? <CorrespondenceSelectRow /> : <></>}
                {props.showPyTorchCluster ? <PyTorchClusterSelectRow /> : <></>}
                {props.showFeature ? <FeatureSelectRow /> : <></>}
                {props.showIndex ? <IndexSelectRow /> : <></>}
                {props.showHalfPrecision ? <HalfPrecisionRow /> : <></>}
                {props.showDefaultTune ? <DefaultTuneRow /> : <></>}

                <ModelUploadButtonRow />
            </>
        )
    }, [guiState.showPyTorchModelUpload])

    return modelUploaderRow
}
