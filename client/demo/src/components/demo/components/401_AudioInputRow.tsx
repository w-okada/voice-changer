import React, { useMemo, useEffect } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"
import { AudioInputMediaRow } from "./401-1_AudioInputMediaRow"

export type AudioInputRowProps = {
}

export const AudioInputRow = (_props: AudioInputRowProps) => {
    const appState = useAppState()
    const guiState = useGuiState()

    // キャッシュの設定は反映（たぶん、設定操作の時も起動していしまう。が問題は起こらないはず）
    useEffect(() => {
        if (typeof appState.clientSetting.clientSetting.audioInput == "string") {
            if (guiState.inputAudioDeviceInfo.find(x => {
                // console.log("COMPARE:", x.deviceId, appState.clientSetting.setting.audioInput)
                return x.deviceId == appState.clientSetting.clientSetting.audioInput
            })) {
                guiState.setAudioInputForGUI(appState.clientSetting.clientSetting.audioInput)
            }
        }
    }, [guiState.inputAudioDeviceInfo, appState.clientSetting.clientSetting.audioInput])


    const audioInputRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1  guided">
                <div className="body-item-title left-padding-1">AudioInput</div>
                <div className="body-select-container">
                    <select className="body-select" value={guiState.audioInputForGUI} onChange={(e) => {
                        guiState.setAudioInputForGUI(e.target.value)
                        if (guiState.audioInputForGUI != "file") {
                            appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, audioInput: e.target.value })
                        }
                    }}>
                        {
                            guiState.inputAudioDeviceInfo.map(x => {
                                return <option key={x.deviceId} value={x.deviceId}>{x.label}</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [guiState.inputAudioDeviceInfo, guiState.audioInputForGUI, appState.clientSetting.clientSetting, appState.clientSetting.updateClientSetting])



    return (
        <>
            {audioInputRow}
            <AudioInputMediaRow />
        </>
    )

}