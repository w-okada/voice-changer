import React, { useMemo } from "react"
import { AnimationTypes, HeaderButton, HeaderButtonProps } from "../101_HeaderButton"
import { useGuiState } from "./001_GuiStateProvider"
import { ModelUploaderRow } from "./301_ModelUploaderRow"
import { ConfigSelectRow } from "./302_ConfigSelectRow"
import { ONNXSelectRow } from "./303_ONNXSelectRow"
import { PyTorchSelectRow } from "./304_PyTorchSelectRow"
import { CorrespondenceSelectRow } from "./305_CorrespondenceSelectRow"
import { PyTorchClusterSelectRow } from "./306_PyTorchClusterSelectRow"
import { ModelUploadButtonRow } from "./310_ModelUploadButtonRow"
import { FrameworkRow } from "./320_FrameworkRow"
import { ONNXExecutorRow } from "./321_ONNXExecutorRow"


export const ModelSetting = () => {
    const guiState = useGuiState()

    const accodionButton = useMemo(() => {
        const accodionButtonProps: HeaderButtonProps = {
            stateControlCheckbox: guiState.stateControls.openModelSettingCheckbox,
            tooltip: "Open/Close",
            onIcon: ["fas", "caret-up"],
            offIcon: ["fas", "caret-up"],
            animation: AnimationTypes.spinner,
            tooltipClass: "tooltip-right",
        };
        return <HeaderButton {...accodionButtonProps}></HeaderButton>;
    }, []);

    const modelSetting = useMemo(() => {

        return (
            <>
                {guiState.stateControls.openModelSettingCheckbox.trigger}
                <div className="partition">
                    <div className="partition-header">
                        <span className="caret">
                            {accodionButton}
                        </span>
                        <span className="title" onClick={() => { guiState.stateControls.openModelSettingCheckbox.updateState(!guiState.stateControls.openModelSettingCheckbox.checked()) }}>
                            Model Setting
                        </span>
                        <span></span>
                    </div>

                    <div className="partition-content">
                        <ModelUploaderRow />
                        <ConfigSelectRow />
                        <ONNXSelectRow />
                        <PyTorchSelectRow />
                        <CorrespondenceSelectRow />
                        <PyTorchClusterSelectRow />
                        <ModelUploadButtonRow />
                        <FrameworkRow />
                        <ONNXExecutorRow />
                    </div>
                </div>
            </>
        )
    }, [])

    return modelSetting
}