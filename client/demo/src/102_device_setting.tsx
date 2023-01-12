import { fileSelectorAsDataURL, createDummyMediaStream, SampleRate } from "@dannadori/voice-changer-client-js"
import React, { useEffect, useMemo, useState } from "react"
import { AUDIO_ELEMENT_FOR_PLAY_RESULT, AUDIO_ELEMENT_FOR_TEST_CONVERTED, AUDIO_ELEMENT_FOR_TEST_ORIGINAL } from "./const"
import { ClientState } from "./hooks/useClient";


const reloadDevices = async () => {
    try {
        await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    } catch (e) {
        console.warn("Enumerate device error::", e)
    }
    const mediaDeviceInfos = await navigator.mediaDevices.enumerateDevices();

    const audioInputs = mediaDeviceInfos.filter(x => { return x.kind == "audioinput" })
    audioInputs.push({
        deviceId: "none",
        groupId: "none",
        kind: "audioinput",
        label: "none",
        toJSON: () => { }
    })
    audioInputs.push({
        deviceId: "file",
        groupId: "file",
        kind: "audioinput",
        label: "file",
        toJSON: () => { }
    })
    const audioOutputs = mediaDeviceInfos.filter(x => { return x.kind == "audiooutput" })
    return [audioInputs, audioOutputs]
}
export type UseDeviceSettingProps = {
    clientState: ClientState
}

export type DeviceSettingState = {
    deviceSetting: JSX.Element;
}

export const useDeviceSetting = (audioContext: AudioContext | null, props: UseDeviceSettingProps): DeviceSettingState => {
    const [inputAudioDeviceInfo, setInputAudioDeviceInfo] = useState<MediaDeviceInfo[]>([])
    const [outputAudioDeviceInfo, setOutputAudioDeviceInfo] = useState<MediaDeviceInfo[]>([])

    const [audioInputForGUI, setAudioInputForGUI] = useState<string>("none")
    const [audioOutputForGUI, setAudioOutputForGUI] = useState<string>("none")


    useEffect(() => {
        const initialize = async () => {
            const audioInfo = await reloadDevices()
            setInputAudioDeviceInfo(audioInfo[0])
            setOutputAudioDeviceInfo(audioInfo[1])
            // if (CHROME_EXTENSION) {
            //     //@ts-ignore
            //     const storedOptions = await chrome.storage.local.get("microphoneOptions")
            //     if (storedOptions) {
            //         setOptions(storedOptions)
            //     }
            // }
        }
        initialize()
    }, [])


    const audioInputRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1  guided">
                <div className="body-item-title left-padding-1">AudioInput</div>
                <div className="body-select-container">
                    <select className="body-select" value={audioInputForGUI} onChange={(e) => {
                        setAudioInputForGUI(e.target.value)
                    }}>
                        {
                            inputAudioDeviceInfo.map(x => {
                                return <option key={x.deviceId} value={x.deviceId}>{x.label}</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [inputAudioDeviceInfo, audioInputForGUI])


    useEffect(() => {
        if (!audioContext) {
            return
        }

        if (audioInputForGUI == "none") {
            const ms = createDummyMediaStream(audioContext)
            props.clientState.clientSetting.setAudioInput(ms)
        } else if (audioInputForGUI == "file") {
            // file selector (audioMediaInputRow)
        } else {
            props.clientState.clientSetting.setAudioInput(audioInputForGUI)
        }
    }, [audioContext, audioInputForGUI, props.clientState.clientSetting.setAudioInput])

    const audioMediaInputRow = useMemo(() => {
        if (audioInputForGUI != "file") {
            return <></>
        }

        const onFileLoadClicked = async () => {
            const url = await fileSelectorAsDataURL("")

            // input stream for client.
            const audio = document.getElementById(AUDIO_ELEMENT_FOR_TEST_CONVERTED) as HTMLAudioElement
            audio.src = url
            await audio.play()
            const src = audioContext!.createMediaElementSource(audio);
            const dst = audioContext!.createMediaStreamDestination()
            src.connect(dst)
            props.clientState.clientSetting.setAudioInput(dst.stream)
            // original stream to play.
            const audio_org = document.getElementById(AUDIO_ELEMENT_FOR_TEST_ORIGINAL) as HTMLAudioElement
            audio_org.src = url
            audio_org.pause()

            // audio_org.onplay = () => {
            //     console.log(audioOutputRef.current)
            //     // @ts-ignore
            //     audio_org.setSinkId(audioOutputRef.current)
            // }
        }

        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title"></div>
                <div className="body-item-text">
                    <div>
                        org:<audio id={AUDIO_ELEMENT_FOR_TEST_ORIGINAL} controls></audio>
                    </div>
                    <div>
                        cnv:<audio id={AUDIO_ELEMENT_FOR_TEST_CONVERTED} controls></audio>
                    </div>
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onFileLoadClicked}>load</div>
                </div>
            </div>
        )
    }, [audioInputForGUI, props.clientState.clientSetting.setAudioInput])



    const audioOutputRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">AudioOutput</div>
                <div className="body-select-container">
                    <select className="body-select" value={audioOutputForGUI} onChange={(e) => { setAudioOutputForGUI(e.target.value) }}>
                        {
                            outputAudioDeviceInfo.map(x => {
                                return <option key={x.deviceId} value={x.deviceId}>{x.label}</option>
                            })
                        }
                    </select>
                    <audio hidden id={AUDIO_ELEMENT_FOR_PLAY_RESULT}></audio>
                </div>
            </div>
        )
    }, [outputAudioDeviceInfo, audioOutputForGUI])

    useEffect(() => {
        if (audioOutputForGUI == "none") return
        [AUDIO_ELEMENT_FOR_PLAY_RESULT, AUDIO_ELEMENT_FOR_TEST_ORIGINAL].forEach(x => {
            const audio = document.getElementById(x) as HTMLAudioElement
            if (audio) {
                // @ts-ignore
                audio.setSinkId(audioOutputForGUI)
            }
        })
    }, [audioOutputForGUI])


    const deviceSetting = useMemo(() => {
        return (
            <>
                <div className="body-row split-3-7 left-padding-1">
                    <div className="body-sub-section-title">Device Setting</div>
                    <div className="body-select-container">
                    </div>
                </div>
                {audioInputRow}
                {audioMediaInputRow}
                {audioOutputRow}
            </>
        )
    }, [audioInputRow, audioMediaInputRow, audioOutputRow])

    return {
        deviceSetting,
    }
}
