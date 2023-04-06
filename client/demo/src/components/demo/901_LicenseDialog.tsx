import React, { useMemo } from "react";
import { useAppRoot } from "../../001_provider/001_AppRootProvider";
import { useGuiState } from "./001_GuiStateProvider";


export const LicenseDialog = () => {
    const { appGuiSettingState } = useAppRoot()
    const guiState = useGuiState()
    const licenses = appGuiSettingState.appGuiSetting.dialogs.license

    const dialog = useMemo(() => {
        const closeButtonRow = (
            <div className="body-row split-3-4-3 left-padding-1">
                <div className="body-item-text">
                </div>
                <div className="body-button-container body-button-container-space-around">
                    <div className="body-button" onClick={() => { guiState.stateControls.showLicenseCheckbox.updateState(false) }} >close</div>
                </div>
                <div className="body-item-text"></div>
            </div>
        )
        const records = licenses.map(x => {
            return (
                <div key={x.url} className="body-row split-3-4-3 left-padding-1">
                    <div className="body-item-text">
                        <a href={x.url} target="_blank" rel="noopener noreferrer">{x.title}</a>
                    </div>
                    <div className="body-item-text">
                        <a href={x.url} target="_blank" rel="noopener noreferrer">{x.auther}({x.contact})</a>
                    </div>
                    <div className="body-item-text">{x.license}</div>
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
    }, [licenses]);
    return dialog;

    return <></>
};
