import React, { useMemo } from "react";
import { useGuiState } from "./001_GuiStateProvider";
import { useAppState } from "../../001_provider/001_AppStateProvider";
import { useAppRoot } from "../../001_provider/001_AppRootProvider";

export const EnablePassThroughDialog = () => {
    const guiState = useGuiState();
    const { audioContextState } = useAppRoot();
    const { serverSetting } = useAppState();
    const { setting } = useAppState();
    const dialog = useMemo(() => {
        const buttonRow = (
            <div className="body-row split-3-4-3 left-padding-1">
                <div className="body-item-text"></div>
                <div className="body-button-container body-button-container-space-around">
                    <div
                        className="body-button"
                        onClick={() => {
                            serverSetting.updateServerSettings({ ...serverSetting.serverSetting, passThrough: true });
                            guiState.stateControls.showEnablePassThroughDialogCheckbox.updateState(false);
                        }}
                    >
                        OK
                    </div>
                    <div
                        className="body-button"
                        onClick={() => {
                            guiState.stateControls.showEnablePassThroughDialogCheckbox.updateState(false);
                        }}
                    >
                        Cancel
                    </div>
                </div>
                <div className="body-item-text"></div>
            </div>
        );

        console.log("AUDIO_CONTEXT", audioContextState.audioContext);
        return (
            <div className="dialog-frame">
                <div className="dialog-title">Enable Pass Through</div>
                <div className="dialog-content">{buttonRow}</div>
            </div>
        );
    }, [setting, audioContextState]);
    return dialog;
};
