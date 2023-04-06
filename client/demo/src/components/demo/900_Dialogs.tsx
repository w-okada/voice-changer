import React from "react";
import { useGuiState } from "./001_GuiStateProvider";
import { LicenseDialog } from "./901_LicenseDialog";

export const Dialogs = () => {
    const guiState = useGuiState()
    const dialogs = (
        <div>
            {guiState.stateControls.showLicenseCheckbox.trigger}
            <div className="dialog-container" id="dialog">
                {guiState.stateControls.showLicenseCheckbox.trigger}
                <LicenseDialog></LicenseDialog>
            </div>
        </div>
    );

    return dialogs
}
