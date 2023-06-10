import React from "react";
import { useGuiState } from "./001_GuiStateProvider";
import { TextInputDialog } from "./911_TextInputDialog";

export const Dialogs2 = () => {
    const guiState = useGuiState()
    const dialogs = (
        <div>
            {guiState.stateControls.showTextInputCheckbox.trigger}
            <div className="dialog-container2" id="dialog2">
                {guiState.stateControls.showTextInputCheckbox.trigger}
                <TextInputDialog></TextInputDialog>
            </div>

        </div>
    );

    return dialogs
}
