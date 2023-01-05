import * as React from "react";
import { createRoot } from "react-dom/client";
import "./css/App.css"
import { useEffect, useMemo, useRef, useState } from "react";
import { VoiceChnagerClient, createDummyMediaStream } from "@dannadori/voice-changer-client-js"
import { useMicrophoneOptions } from "./options_microphone";
const container = document.getElementById("app")!;
const root = createRoot(container);

const App = () => {


    const audioContextRef = useRef<AudioContext>()
    const voiceChangerClientRef = useRef<VoiceChnagerClient | null>(null)
    const [clientInitialized, setClientInitialized] = useState<boolean>(false)
    const [bufferingTime, setBufferingTime] = useState<number>(0)
    const [responseTime, setResponseTime] = useState<number>(0)
    const [volume, setVolume] = useState<number>(0)

    const { component: microphoneSettingComponent, options: microphoneOptions, params: microphoneParams, isStarted } = useMicrophoneOptions(audioContextRef.current)

    const onClearSettingClicked = async () => {
        //@ts-ignore
        await chrome.storage.local.clear();
        //@ts-ignore
        await chrome.storage.sync.clear();

        location.reload()
    }


    useEffect(() => {
        const initialized = async () => {
            audioContextRef.current = new AudioContext()
            voiceChangerClientRef.current = new VoiceChnagerClient(audioContextRef.current, true, {
                notifySendBufferingTime: (val: number) => {
                    setBufferingTime(val)
                },
                notifyResponseTime: (val: number) => {
                    setResponseTime(val)
                },
                notifyException: (mes: string) => {
                    if (mes.length > 0) {
                        console.log(`error:${mes}`)
                    }
                }
            }, {
                notifyVolume: (vol: number) => {
                    setVolume(vol)
                }
            })
            await voiceChangerClientRef.current.isInitialized()
            setClientInitialized(true)

            const audio = document.getElementById("audio-output") as HTMLAudioElement
            audio.srcObject = voiceChangerClientRef.current.stream
            audio.play()
        }
        initialized()
    }, [])

    useEffect(() => {
        const start = async () => {
            if (!voiceChangerClientRef.current || !clientInitialized) {
                // console.log("client is not initialized")
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
                // console.log("client is not initialized")
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


    useEffect(() => {
        const changeInput = async () => {
            if (!voiceChangerClientRef.current || !clientInitialized) {
                // console.log("client is not initialized")
                return
            }
            if (!microphoneOptions.audioInput || microphoneOptions.audioInput == "none") {
                const ms = createDummyMediaStream(audioContextRef.current!)
                await voiceChangerClientRef.current.setup(ms, microphoneOptions.bufferSize, microphoneOptions.forceVfDisable)

            } else {
                await voiceChangerClientRef.current.setup(microphoneOptions.audioInput, microphoneOptions.bufferSize, microphoneOptions.forceVfDisable)
            }

        }
        changeInput()
    }, [microphoneOptions.audioInput, microphoneOptions.bufferSize, microphoneOptions.forceVfDisable])


    useEffect(() => {
        if (!voiceChangerClientRef.current || !clientInitialized) {
            // console.log("client is not initialized")
            return
        }
        voiceChangerClientRef.current.setInputChunkNum(microphoneOptions.inputChunkNum)
    }, [microphoneOptions.inputChunkNum])

    useEffect(() => {
        if (!voiceChangerClientRef.current || !clientInitialized) {
            // console.log("client is not initialized")
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
    const performanceRow = useMemo(() => {
        return (
            <>
                <div className="body-row split-3-1-1-1-4 left-padding-1 highlight">
                    <div className="body-item-title">monitor:</div>
                    <div className="body-item-text">vol(rms):{volume.toFixed(4)}</div>
                    <div className="body-item-text">buf(ms):{bufferingTime}</div>
                    <div className="body-item-text">res(ms):{responseTime}</div>
                    <div className="body-item-text"></div>
                </div>
            </>
        )
    }, [volume, bufferingTime, responseTime])
    return (
        <div className="body">
            <div className="body-row">
                <div className="body-top-title">
                    Voice Changer Setting
                </div>
            </div>
            {clearRow}
            {performanceRow}
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
