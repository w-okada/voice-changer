import * as React from "react";
import { createRoot } from "react-dom/client";
import "./css/App.css"
import { useEffect, useMemo, useRef, useState } from "react";
import { VoiceChnagerClient } from "@dannadori/voice-changer-client-js"
import { useMicrophoneOptions } from "./options_microphone";
const container = document.getElementById("app")!;
const root = createRoot(container);

const App = () => {
    const { component: microphoneSettingComponent, options: microphoneOptions, params: microphoneParams, isStarted } = useMicrophoneOptions()

    const voiceChangerClientRef = useRef<VoiceChnagerClient | null>(null)
    const [clientInitialized, setClientInitialized] = useState<boolean>(false)

    const onClearSettingClicked = async () => {
        //@ts-ignore
        await chrome.storage.local.clear();
        //@ts-ignore
        await chrome.storage.sync.clear();

        location.reload()
    }


    useEffect(() => {
        const initialized = async () => {
            const ctx = new AudioContext()
            voiceChangerClientRef.current = new VoiceChnagerClient(ctx, true, {
                notifySendBufferingTime: (val: number) => { console.log(`buf:${val}`) },
                notifyResponseTime: (val: number) => { console.log(`res:${val}`) },
                notifyException: (mes: string) => {
                    if (mes.length > 0) {
                        console.log(`error:${mes}`)
                    }
                }
            }, { notifyVolume: (vol: number) => { } })
            await voiceChangerClientRef.current.isInitialized()
            setClientInitialized(true)

            const audio = document.getElementById("audio-output") as HTMLAudioElement
            audio.srcObject = voiceChangerClientRef.current.stream
            audio.play()
        }
        initialized()
    }, [])

    useEffect(() => {
        console.log("START!!!", isStarted)
        const start = async () => {
            if (!voiceChangerClientRef.current || !clientInitialized) {
                console.log("client is not initialized")
                return
            }
            // if (!microphoneOptions.audioInputDeviceId || microphoneOptions.audioInputDeviceId.length == 0) {
            //     console.log("audioInputDeviceId is not initialized")
            //     return
            // }
            // await voiceChangerClientRef.current.setup(microphoneOptions.audioInputDeviceId!, microphoneOptions.bufferSize)
            voiceChangerClientRef.current.setServerUrl(microphoneOptions.mmvcServerUrl, microphoneOptions.protocol, true)
            voiceChangerClientRef.current.start()
        }
        const stop = async () => {
            if (!voiceChangerClientRef.current || !clientInitialized) {
                console.log("client is not initialized")
                return
            }
            voiceChangerClientRef.current.stop()
        }
        if (isStarted) {
            start()
        } else {
            stop()
        }
    }, [isStarted])

    // useEffect(() => {
    //     if (!voiceChangerClientRef.current || !clientInitialized) {
    //         console.log("client is not initialized")
    //         return
    //     }
    //     voiceChangerClientRef.current.setServerUrl(microphoneOptions.mmvcServerUrl, microphoneOptions.protocol, false)
    // }, [microphoneOptions.mmvcServerUrl, microphoneOptions.protocol])

    useEffect(() => {
        const changeInput = async () => {
            if (!voiceChangerClientRef.current || !clientInitialized) {
                console.log("client is not initialized")
                return
            }
            await voiceChangerClientRef.current.setup(microphoneOptions.audioInputDeviceId!, microphoneOptions.bufferSize, microphoneOptions.forceVfDisable)
        }
        changeInput()
    }, [microphoneOptions.audioInputDeviceId!, microphoneOptions.bufferSize, microphoneOptions.forceVfDisable])


    useEffect(() => {
        if (!voiceChangerClientRef.current || !clientInitialized) {
            console.log("client is not initialized")
            return
        }
        voiceChangerClientRef.current.setInputChunkNum(microphoneOptions.inputChunkNum)
    }, [microphoneOptions.inputChunkNum])

    useEffect(() => {
        if (!voiceChangerClientRef.current || !clientInitialized) {
            console.log("client is not initialized")
            return
        }
        voiceChangerClientRef.current.setVoiceChangerMode(microphoneOptions.voiceChangerMode)
    }, [microphoneOptions.voiceChangerMode])

    useEffect(() => {
        if (!voiceChangerClientRef.current || !clientInitialized) {
            console.log("client is not initialized")
            return
        }
        voiceChangerClientRef.current.setRequestParams(microphoneParams)
    }, [microphoneParams])

    const clearRow = useMemo(() => {
        return (
            <>
                <div className="body-row split-3-3-4 left-padding-1 highlight">
                    <div className="body-item-title">Clear Setting</div>
                    <div className="body-item-text"></div>
                    <div className="body-button-container">
                        <div className="body-button" onClick={onClearSettingClicked}>clear</div>
                    </div>
                </div>
            </>
        )
    }, [])

    return (
        <div className="body">
            <div className="body-row">
                <div className="body-top-title">
                    Voice Changer Setting
                </div>
            </div>
            {clearRow}
            {microphoneSettingComponent}
            <div>
                <audio id="audio-output"></audio>
            </div>
        </div>
    )
}

root.render(
    <App></App>
);
