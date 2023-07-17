import React from "react";
import { useGuiState } from "./001_GuiStateProvider";
import { TextInputDialog } from "./911_TextInputDialog";
import { ShowLicenseDialog } from "./912_ShowLicenseDialog";

export const Dialogs2 = () => {
    const guiState = useGuiState();
    const dialogs = (
        <div>
            {guiState.stateControls.showTextInputCheckbox.trigger}
            {guiState.stateControls.showLicenseCheckbox.trigger}
            <div className="dialog-container2" id="dialog2">
                {guiState.stateControls.showTextInputCheckbox.trigger}
                <TextInputDialog></TextInputDialog>
                {guiState.stateControls.showLicenseCheckbox.trigger}
                <ShowLicenseDialog />
            </div>
        </div>
    );

    return dialogs;
};
