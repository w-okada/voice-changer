import { getLicenceInfo } from "@dannadori/voice-changer-client-js";
import React, { useMemo } from "react";
import { useAppState } from "../001_provider/001_AppStateProvider";

export const LicenseDialog = () => {
    const { frontendManagerState } = useAppState();

    const form = useMemo(() => {
        const closeButtonRow = (
            <div className="body-row split-3-4-3 left-padding-1">
                <div className="body-item-text">
                </div>
                <div className="body-button-container body-button-container-space-around">
                    <div className="body-button" onClick={() => { frontendManagerState.stateControls.showLicenseCheckbox.updateState(false) }} >close</div>
                </div>
                <div className="body-item-text"></div>
            </div>
        )
        const records = getLicenceInfo().map(x => {
            return (
                <div key={x.url} className="body-row split-3-4-3 left-padding-1">
                    <div className="body-item-text">
                        <a href={x.url} target="_blank" rel="noopener noreferrer">{x.name}</a>
                    </div>
                    <div className="body-item-text">
                        <a href={x.licenseUrl} target="_blank" rel="noopener noreferrer">{x.license}</a>
                    </div>
                    <div className="body-item-text"></div>
                </div>
            )
        })
        return (
            <div className="dialog-frame">
                <div className="dialog-title">License</div>
                <div className="dialog-content">
                    <div className={"dialog-application-title"}>Voice Changer Demo</div>
                    {records}
                    {closeButtonRow}
                </div>
            </div>
        );
    }, []);
    return form;
};
