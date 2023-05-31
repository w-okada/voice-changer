import React from "react";
import { useGuiState } from "./001_GuiStateProvider";
import { LicenseDialog } from "./901_LicenseDialog";
import { WaitingDialog } from "./902_WaitingDialog";
import { StartingNoticeDialog } from "./903_StartingNoticeDialog";

export const Dialogs = () => {
    const guiState = useGuiState()
    const dialogs = (
        <div>
            {guiState.stateControls.showLicenseCheckbox.trigger}
            {guiState.stateControls.showWaitingCheckbox.trigger}
            {guiState.stateControls.showStartingNoticeCheckbox.trigger}
            <div className="dialog-container" id="dialog">
                {guiState.stateControls.showLicenseCheckbox.trigger}
                <LicenseDialog></LicenseDialog>
                {guiState.stateControls.showWaitingCheckbox.trigger}
                <WaitingDialog></WaitingDialog>
                {guiState.stateControls.showStartingNoticeCheckbox.trigger}
                <StartingNoticeDialog></StartingNoticeDialog>
            </div>

        </div>
    );

    return dialogs
}
