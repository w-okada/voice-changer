import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"

export type AudioDeviceModeRowProps = {
}

export const AudioDeviceModeRow = (_props: AudioDeviceModeRowProps) => {
    const appState = useAppState()
    const guiState = useGuiState()
    const serverAudioInputRow = useMemo(() => {
        const enableServerAudio = appState.serverSetting.serverSetting.enableServerAudio
        const serverChecked = enableServerAudio == 1 ? true : false
        const clientChecked = enableServerAudio == 1 ? false : true

        const onDeviceModeChanged = (val: number) => {
            if (guiState.isConverting) {
                alert("cannot change mode when voice conversion is enabled")
                return
            }
            appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, enableServerAudio: val })
        }
        return (
            <div className="body-row split-3-3-4 left-padding-1  guided">

                <div className="body-item-title left-padding-1">Device Mode</div>
                <div className="body-input-container">
                    <div className="left-padding-1">
                        <input type="radio" id="client-device" name="device-mode" checked={clientChecked} onChange={() => { onDeviceModeChanged(0) }} />
                        <label htmlFor="client-device">client device</label>
                    </div>
                    <div className="left-padding-1">
                        <input className="left-padding-1" type="radio" id="server-device" name="device-mode" checked={serverChecked} onChange={() => { onDeviceModeChanged(1) }} />
                        <label htmlFor="server-device">server device(exp.)</label>
                    </div>
                </div>
                <div></div>
            </div>
        )
    }, [appState.serverSetting.serverSetting, appState.serverSetting.updateServerSettings, guiState.isConverting])



    return (
        <>
            {serverAudioInputRow}
        </>
    )

}