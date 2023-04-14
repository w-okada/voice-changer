import React, { useMemo } from "react";
import { useGuiState } from "./001_GuiStateProvider";


export const WaitingDialog = () => {
    const guiState = useGuiState()

    const dialog = useMemo(() => {
        const closeButtonRow = (
            <div className="body-row split-3-4-3 left-padding-1">
                <div className="body-item-text">
                </div>
                <div className="body-button-container body-button-container-space-around">
                    <div className="body-button" onClick={() => { guiState.stateControls.showWaitingCheckbox.updateState(false) }} >close</div>
                </div>
                <div className="body-item-text"></div>
            </div>
        )
        const content = (
            <div className="body-row split-3-4-3 left-padding-1">
                <div className="body-item-text">
                </div>
                <div className="body-item-text">
                    please wait... (about 1 min)
                </div>
                <div className="body-item-text"></div>
            </div>
        )

        return (
            <div className="dialog-frame">
                <div className="dialog-title">export onnx file</div>
                <div className="dialog-content">
                    {content}
                    {/* {closeButtonRow} */}
                </div>
            </div>
        );
    }, []);
    return dialog;

};
