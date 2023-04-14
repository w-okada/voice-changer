import React from "react";
import { useGuiState } from "./001_GuiStateProvider";
import { LicenseDialog } from "./901_LicenseDialog";
import { WaitingDialog } from "./902_WaitingDialog";

export const Dialogs = () => {
    const guiState = useGuiState()
    const dialogs = (
        <div>
            {guiState.stateControls.showLicenseCheckbox.trigger}
            {guiState.stateControls.showWaitingCheckbox.trigger}
            <div className="dialog-container" id="dialog">
                {guiState.stateControls.showLicenseCheckbox.trigger}
                <LicenseDialog></LicenseDialog>
                {guiState.stateControls.showWaitingCheckbox.trigger}
                <WaitingDialog></WaitingDialog>
            </div>

        </div>
    );

    return dialogs
}
