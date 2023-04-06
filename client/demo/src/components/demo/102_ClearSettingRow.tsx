import React, { useMemo } from "react";
import { useAppState } from "../../001_provider/001_AppStateProvider";
import { useIndexedDB } from "@dannadori/voice-changer-client-js";
import { INDEXEDDB_KEY_AUDIO_OUTPUT } from "../../const";
import { useAppRoot } from "../../001_provider/001_AppRootProvider";

export const ClearSettingRow = () => {
    const appState = useAppState()
    const { appGuiSettingState } = useAppRoot()
    const clientType = appGuiSettingState.appGuiSetting.id
    const { removeItem } = useIndexedDB({ clientType: clientType })


    const clearSettingRow = useMemo(() => {
        const onClearSettingClicked = async () => {
            await appState.clearSetting()
            await removeItem(INDEXEDDB_KEY_AUDIO_OUTPUT)
            location.reload()
        }
        return (
            <div className="body-row split-3-3-4 left-padding-1">
                <div className="body-button-container">
                    <div className="body-button" onClick={onClearSettingClicked}>clear setting</div>
                </div>
                <div className="body-item-text"></div>
                <div className="body-item-text"></div>
            </div>
        )
    }, [])
    return clearSettingRow
};
