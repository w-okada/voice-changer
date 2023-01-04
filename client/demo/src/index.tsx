import * as React from "react";
import { createRoot } from "react-dom/client";
import "./css/App.css"
import { useEffect, useMemo, useRef } from "react";
import { VoiceChnagerClient } from "@dannadori/voice-changer-client-js"
import { useMicrophoneOptions } from "./options_microphone";
const container = document.getElementById("app")!;
const root = createRoot(container);

const App = () => {
    const { component: microphoneSettingComponent, options: microphonOptions } = useMicrophoneOptions()

    const voiceChnagerClientRef = useRef<VoiceChnagerClient | null>(null)

    console.log(microphonOptions)

    const onClearSettingClicked = async () => {
        //@ts-ignore
        await chrome.storage.local.clear();
        //@ts-ignore
        await chrome.storage.sync.clear();

        location.reload()
    }

    useEffect(() => {
        if (microphonOptions.audioInputDeviceId.length == 0) {
            return
        }
        const setAudio = async () => {
            const ctx = new AudioContext()

            if (voiceChnagerClientRef.current) {

            }
            voiceChnagerClientRef.current = new VoiceChnagerClient(ctx, true, {
                notifySendBufferingTime: (val: number) => { console.log(`buf:${val}`) },
                notifyResponseTime: (val: number) => { console.log(`res:${val}`) },
                notifyException: (mes: string) => { console.log(`error:${mes}`) }
            })
            await voiceChnagerClientRef.current.isInitialized()

            voiceChnagerClientRef.current.setServerUrl("https://192.168.0.3:18888/test", "sio")
            voiceChnagerClientRef.current.setup(microphonOptions.audioInputDeviceId, 1024)

            const audio = document.getElementById("audio-output") as HTMLAudioElement
            audio.srcObject = voiceChnagerClientRef.current.stream
            audio.play()
        }
        setAudio()
    }, [microphonOptions.audioInputDeviceId])

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
