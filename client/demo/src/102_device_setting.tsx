import { fileSelectorAsDataURL, useIndexedDB } from "@dannadori/voice-changer-client-js"
import React, { useEffect, useMemo, useRef, useState } from "react"
import { AUDIO_ELEMENT_FOR_PLAY_RESULT, AUDIO_ELEMENT_FOR_TEST_CONVERTED, AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK, AUDIO_ELEMENT_FOR_TEST_ORIGINAL, INDEXEDDB_KEY_AUDIO_OUTPUT } from "./const"
import { ClientState } from "@dannadori/voice-changer-client-js";


const reloadDevices = async () => {
    try {
        const ms = await navigator.mediaDevices.getUserMedia({ video: false, audio: true });
        ms.getTracks().forEach(x => { x.stop() })
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
    const [fileInputEchoback, setFileInputEchoback] = useState<boolean>()//最初のmuteが有効になるように。undefined
    const { getItem, setItem } = useIndexedDB()

    const audioSrcNode = useRef<MediaElementAudioSourceNode>()

    useEffect(() => {
        const initialize = async () => {
            const audioInfo = await reloadDevices()
            setInputAudioDeviceInfo(audioInfo[0])
            setOutputAudioDeviceInfo(audioInfo[1])
        }
        initialize()
    }, [])

    useEffect(() => {
        if (typeof props.clientState.clientSetting.setting.audioInput == "string") {
            if (inputAudioDeviceInfo.find(x => {
                // console.log("COMPARE:", x.deviceId, props.clientState.clientSetting.setting.audioInput)
                return x.deviceId == props.clientState.clientSetting.setting.audioInput
            })) {
                setAudioInputForGUI(props.clientState.clientSetting.setting.audioInput)

            }
        }
    }, [inputAudioDeviceInfo, props.clientState.clientSetting.setting.audioInput])

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
    }, [inputAudioDeviceInfo, audioInputForGUI, props.clientState.clientSetting.setting.audioInput])


    useEffect(() => {
        if (!audioContext) {
            return
        }

        if (audioInputForGUI == "file") {
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
            audio.pause()
            audio.srcObject = null
            audio.src = url
            await audio.play()
            if (!audioSrcNode.current) {
                audioSrcNode.current = audioContext!.createMediaElementSource(audio);
            }
            if (audioSrcNode.current.mediaElement != audio) {
                audioSrcNode.current = audioContext!.createMediaElementSource(audio);
            }

            const dst = audioContext!.createMediaStreamDestination()
            audioSrcNode.current.connect(dst)
            props.clientState.clientSetting.setAudioInput(dst.stream)

            const audio_echo = document.getElementById(AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK) as HTMLAudioElement
            audio_echo.srcObject = dst.stream
            audio_echo.play()
            setFileInputEchoback(false)

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
                    <div style={{ display: "none" }}>
                        org:<audio id={AUDIO_ELEMENT_FOR_TEST_ORIGINAL} controls></audio>
                    </div>
                    <div>
                        <audio id={AUDIO_ELEMENT_FOR_TEST_CONVERTED} controls></audio>
                        <audio id={AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK} controls hidden></audio>
                    </div>
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onFileLoadClicked}>load</div>
                    <input type="checkbox" checked={fileInputEchoback} onChange={(e) => { setFileInputEchoback(e.target.checked) }} /> echoback
                </div>
            </div>
        )
    }, [audioInputForGUI, props.clientState.clientSetting.setAudioInput, fileInputEchoback])



    const audioOutputRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">AudioOutput</div>
                <div className="body-select-container">
                    <select className="body-select" value={audioOutputForGUI} onChange={(e) => {
                        setAudioOutputForGUI(e.target.value)
                        setItem(INDEXEDDB_KEY_AUDIO_OUTPUT, e.target.value)
                    }}>
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
        [AUDIO_ELEMENT_FOR_PLAY_RESULT, AUDIO_ELEMENT_FOR_TEST_ORIGINAL, AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK].forEach(x => {
            const audio = document.getElementById(x) as HTMLAudioElement
            if (audio) {
                if (audioOutputForGUI == "none") {
                    // @ts-ignore
                    audio.setSinkId("")
                    if (x == AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK) {
                        audio.volume = fileInputEchoback ? 1 : 0
                    }
                } else {
                    // @ts-ignore
                    audio.setSinkId(audioOutputForGUI)
                    if (x == AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK) {
                        audio.volume = fileInputEchoback ? 1 : 0
                    }
                }
            }
        })
    }, [audioOutputForGUI, audioInputForGUI])


    useEffect(() => {
        const loadCache = async () => {
            const key = await getItem(INDEXEDDB_KEY_AUDIO_OUTPUT)
            if (key) {
                setAudioOutputForGUI(key as string)
            }
        }
        loadCache()
    }, [])


    useEffect(() => {
        [AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK].forEach(x => {
            const audio = document.getElementById(x) as HTMLAudioElement
            if (audio) {
                audio.volume = fileInputEchoback ? 1 : 0
            }
        })
    }, [fileInputEchoback])

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
