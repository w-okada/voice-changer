import React, { useMemo, useEffect, useState } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"
import { AudioInputMediaRow } from "./401-1_AudioInputMediaRow"

export type AudioInputRowProps = {
}

export const AudioInputRow = (_props: AudioInputRowProps) => {
    const appState = useAppState()
    const guiState = useGuiState()
    const [hostApi, setHostApi] = useState<string>("")

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
        if (appState.serverSetting.serverSetting.enableServerAudio == 1) {
            return <></>
        }

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
    }, [guiState.inputAudioDeviceInfo, guiState.audioInputForGUI, appState.clientSetting.clientSetting, appState.clientSetting.updateClientSetting, appState.serverSetting.serverSetting.enableServerAudio])


    const serverAudioInputRow = useMemo(() => {
        if (appState.serverSetting.serverSetting.enableServerAudio == 0) {
            return <></>
        }
        const devices = appState.serverSetting.serverSetting.serverAudioInputDevices
        const hostAPIs = new Set(devices.map(x => { return x.hostAPI }))
        const hostAPIOptions = Array.from(hostAPIs).map((x, index) => { return <option value={x} key={index} >{x}</option> })

        const filteredDevice = devices.map((x, index) => {
            const className = (x.hostAPI == hostApi || hostApi == "") ? "select-option-red" : ""
            return <option className={className} value={x.index} key={index}>[{x.hostAPI}]{x.name}</option>
        })


        return (
            <div className="body-row split-3-7 left-padding-1  guided">
                <div className="body-item-title left-padding-1">AudioInput</div>
                <div className="body-select-container">
                    <div className="body-select-container">
                        <select name="kinds" id="kinds" value={hostApi} onChange={(e) => { setHostApi(e.target.value) }}>
                            {hostAPIOptions}
                        </select>
                        <select className="body-select" value={appState.serverSetting.serverSetting.serverInputDeviceId} onChange={(e) => {
                            appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, serverInputDeviceId: Number(e.target.value) })

                        }}>
                            {filteredDevice}
                        </select>
                    </div>
                </div>
            </div>
        )
    }, [hostApi, appState.serverSetting.serverSetting, appState.serverSetting.updateServerSettings])


    return (
        <>
            {audioInputRow}
            <AudioInputMediaRow />
            {serverAudioInputRow}
        </>
    )

}