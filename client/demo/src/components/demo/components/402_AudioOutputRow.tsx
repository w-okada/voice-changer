import React, { useMemo, useEffect, useState } from "react"
import { useIndexedDB } from "@dannadori/voice-changer-client-js"
import { AudioOutputRecordRow } from "./402-1_AudioOutputRecordRow"
import { useGuiState } from "../001_GuiStateProvider"
import { useAppRoot } from "../../../001_provider/001_AppRootProvider"
import { AUDIO_ELEMENT_FOR_PLAY_RESULT, AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK, AUDIO_ELEMENT_FOR_TEST_ORIGINAL, INDEXEDDB_KEY_AUDIO_OUTPUT } from "../../../const"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export type AudioOutputRowProps = {
}

export const AudioOutputRow = (_props: AudioOutputRowProps) => {
    const { setAudioOutputElementId, initializedRef } = useAppState()
    const appState = useAppState()
    const guiState = useGuiState()
    const { appGuiSettingState } = useAppRoot()
    const clientType = appGuiSettingState.appGuiSetting.id
    const { getItem, setItem } = useIndexedDB({ clientType: clientType })
    const [hostApi, setHostApi] = useState<string>("")

    useEffect(() => {
        const loadCache = async () => {
            const key = await getItem(INDEXEDDB_KEY_AUDIO_OUTPUT)
            if (key) {
                guiState.setAudioOutputForGUI(key as string)
            }
        }
        loadCache()
    }, [])

    useEffect(() => {
        const setAudioOutput = async () => {
            const mediaDeviceInfos = await navigator.mediaDevices.enumerateDevices();

            [AUDIO_ELEMENT_FOR_PLAY_RESULT, AUDIO_ELEMENT_FOR_TEST_ORIGINAL, AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK].forEach(x => {
                const audio = document.getElementById(x) as HTMLAudioElement
                if (audio) {
                    if (appState.serverSetting.serverSetting.enableServerAudio == 1) {

                        // Server Audio を使う場合はElementから音は出さない。
                        audio.volume = 0
                    } else if (guiState.audioOutputForGUI == "none") {
                        // @ts-ignore
                        audio.setSinkId("")
                        if (x == AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK) {
                            audio.volume = 0
                        } else {
                            audio.volume = 0
                        }
                    } else {
                        const audioOutputs = mediaDeviceInfos.filter(x => { return x.kind == "audiooutput" })
                        const found = audioOutputs.some(x => { return x.deviceId == guiState.audioOutputForGUI })
                        if (found) {
                            // @ts-ignore // 例外キャッチできないので事前にIDチェックが必要らしい。！？
                            audio.setSinkId(guiState.audioOutputForGUI)
                        } else {
                            console.warn("No audio output device. use default")
                        }

                        if (x == AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK) {
                            audio.volume = guiState.fileInputEchoback ? 1 : 0
                        } else {
                            audio.volume = 1
                        }
                    }
                }
            })
        }
        setAudioOutput()
    }, [guiState.audioOutputForGUI, guiState.fileInputEchoback, appState.serverSetting.serverSetting.enableServerAudio])



    const audioOutputRow = useMemo(() => {
        if (appState.serverSetting.serverSetting.enableServerAudio == 1) {
            return <></>
        }

        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">AudioOutput</div>
                <div className="body-select-container">
                    <select className="body-select" value={guiState.audioOutputForGUI} onChange={(e) => {
                        guiState.setAudioOutputForGUI(e.target.value)
                        setItem(INDEXEDDB_KEY_AUDIO_OUTPUT, e.target.value)
                    }}>
                        {
                            guiState.outputAudioDeviceInfo.map(x => {
                                return <option key={x.deviceId} value={x.deviceId}>{x.label}</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [appState.serverSetting.serverSetting.enableServerAudio, guiState.outputAudioDeviceInfo, guiState.audioOutputForGUI])

    useEffect(() => {
        setAudioOutputElementId(AUDIO_ELEMENT_FOR_PLAY_RESULT)
    }, [initializedRef.current])


    const serverAudioOutputRow = useMemo(() => {
        if (appState.serverSetting.serverSetting.enableServerAudio == 0) {
            return <></>
        }
        const devices = appState.serverSetting.serverSetting.serverAudioOutputDevices
        const hostAPIs = new Set(devices.map(x => { return x.hostAPI }))
        const hostAPIOptions = Array.from(hostAPIs).map((x, index) => { return <option value={x} key={index} >{x}</option> })

        // const filteredDevice = devices.filter(x => { return x.hostAPI == hostApi || hostApi == "" }).map((x, index) => { return <option value={x.index} key={index}>{x.name}</option> })
        const filteredDevice = devices.map((x, index) => {
            const className = (x.hostAPI == hostApi || hostApi == "") ? "select-option-red" : ""
            return <option className={className} value={x.index} key={index}>[{x.hostAPI}]{x.name}</option>

        })

        return (
            <div className="body-row split-3-7 left-padding-1  guided">
                <div className="body-item-title left-padding-1">AudioOutput</div>
                <div className="body-select-container">
                    <div className="body-select-container">
                        <select name="kinds" id="kinds" value={hostApi} onChange={(e) => { setHostApi(e.target.value) }}>
                            {hostAPIOptions}
                        </select>
                        <select className="body-select" value={appState.serverSetting.serverSetting.serverOutputDeviceId} onChange={(e) => {
                            appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, serverOutputDeviceId: Number(e.target.value) })
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
            {audioOutputRow}
            {appState.serverSetting.serverSetting.enableServerAudio == 0 ? <AudioOutputRecordRow /> : <></>}

            {serverAudioOutputRow}
            <audio hidden id={AUDIO_ELEMENT_FOR_PLAY_RESULT}></audio>
        </>
    )
}
