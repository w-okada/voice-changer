import React from "react";
import { useAppState } from "../001_provider/001_AppStateProvider";
import { LicenseDialog } from "./202_LicenseDialog";

export const Dialog = () => {
    const { frontendManagerState } = useAppState();

    return (
        <div>
            {frontendManagerState.stateControls.showLicenseCheckbox.trigger}
            <div className="dialog-container" id="dialog">
                {frontendManagerState.stateControls.showLicenseCheckbox.trigger}
                <LicenseDialog></LicenseDialog>
            </div>
        </div>
    );
};
