import React, { useMemo, useEffect } from "react"
import { useGuiState } from "./001_GuiStateProvider"
import { useIndexedDB } from "@dannadori/voice-changer-client-js"
import { useAppState } from "../../001_provider/001_AppStateProvider"
import { AUDIO_ELEMENT_FOR_PLAY_RESULT, AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK, AUDIO_ELEMENT_FOR_TEST_ORIGINAL, INDEXEDDB_KEY_AUDIO_OUTPUT } from "../../const"

export const AudioOutputRow = () => {
    const guiState = useGuiState()
    const appState = useAppState()
    const clientType = appState.appGuiSettingState.appGuiSetting.id
    const { getItem, setItem } = useIndexedDB({ clientType: clientType })


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
                    if (guiState.audioOutputForGUI == "none") {
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
    }, [guiState.audioOutputForGUI, guiState.fileInputEchoback])



    const audioOutputRow = useMemo(() => {
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
                    <audio hidden id={AUDIO_ELEMENT_FOR_PLAY_RESULT}></audio>
                </div>
            </div>
        )
    }, [guiState.outputAudioDeviceInfo, guiState.audioOutputForGUI])

    return audioOutputRow
}