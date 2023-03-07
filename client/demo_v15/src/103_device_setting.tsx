import { fileSelectorAsDataURL, useIndexedDB } from "@dannadori/voice-changer-client-js"
import React, { useEffect, useMemo, useRef, useState } from "react"
import { AUDIO_ELEMENT_FOR_PLAY_RESULT, AUDIO_ELEMENT_FOR_TEST_CONVERTED, AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK, AUDIO_ELEMENT_FOR_TEST_ORIGINAL, INDEXEDDB_KEY_AUDIO_OUTPUT } from "./const"
import { useAppState } from "./001_provider/001_AppStateProvider";
import { AnimationTypes, HeaderButton, HeaderButtonProps } from "./components/101_HeaderButton";


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
    audioOutputs.push({
        deviceId: "none",
        groupId: "none",
        kind: "audiooutput",
        label: "none",
        toJSON: () => { }
    })
    // audioOutputs.push({
    //     deviceId: "record",
    //     groupId: "record",
    //     kind: "audiooutput",
    //     label: "record",
    //     toJSON: () => { }
    // })
    return [audioInputs, audioOutputs]
}

export type DeviceSettingState = {
    deviceSetting: JSX.Element;
}

export const useDeviceSetting = (): DeviceSettingState => {
    const appState = useAppState()
    const accodionButton = useMemo(() => {
        const accodionButtonProps: HeaderButtonProps = {
            stateControlCheckbox: appState.frontendManagerState.stateControls.openDeviceSettingCheckbox,
            tooltip: "Open/Close",
            onIcon: ["fas", "caret-up"],
            offIcon: ["fas", "caret-up"],
            animation: AnimationTypes.spinner,
            tooltipClass: "tooltip-right",
        };
        return <HeaderButton {...accodionButtonProps}></HeaderButton>;
    }, []);

    const [inputAudioDeviceInfo, setInputAudioDeviceInfo] = useState<MediaDeviceInfo[]>([])
    const [outputAudioDeviceInfo, setOutputAudioDeviceInfo] = useState<MediaDeviceInfo[]>([])

    const [audioInputForGUI, setAudioInputForGUI] = useState<string>("none")
    const [audioOutputForGUI, setAudioOutputForGUI] = useState<string>("none")
    const [fileInputEchoback, setFileInputEchoback] = useState<boolean>()//最初のmuteが有効になるように。undefined
    const { getItem, setItem } = useIndexedDB()

    const audioSrcNode = useRef<MediaElementAudioSourceNode>()

    const [outputRecordingStarted, setOutputRecordingStarted] = useState<boolean>(false)

    const [useServerMicrophone, setUseServerMicrophone] = useState<boolean>(false)

    // リスト内の
    useEffect(() => {
        const initialize = async () => {
            const audioInfo = await reloadDevices()
            setInputAudioDeviceInfo(audioInfo[0])
            setOutputAudioDeviceInfo(audioInfo[1])
            // if (useServerMicrophone) {
            //     try {
            //         const serverDevices = await appState.serverSetting.getServerDevices()
            //         setServerInputAudioDeviceInfo(serverDevices.audio_input_devices)
            //     } catch (e) {
            //         console.warn(e)
            //     }
            // }
        }
        initialize()
    }, [useServerMicrophone])

    // キャッシュの設定は反映（たぶん、設定操作の時も起動していしまう。が問題は起こらないはず）
    useEffect(() => {
        if (typeof appState.clientSetting.clientSetting.audioInput == "string") {
            if (inputAudioDeviceInfo.find(x => {
                // console.log("COMPARE:", x.deviceId, appState.clientSetting.setting.audioInput)
                return x.deviceId == appState.clientSetting.clientSetting.audioInput
            })) {
                setAudioInputForGUI(appState.clientSetting.clientSetting.audioInput)
            }
        }
    }, [inputAudioDeviceInfo, appState.clientSetting.clientSetting.audioInput])

    const audioInputRow = useMemo(() => {
        if (useServerMicrophone) {
            return <></>
        }
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
    }, [inputAudioDeviceInfo, audioInputForGUI, useServerMicrophone])

    useEffect(() => {
        if (audioInputForGUI == "file") {
            // file selector (audioMediaInputRow)
        } else {
            if (!useServerMicrophone) {
                appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, audioInput: audioInputForGUI })
            } else {
                console.log("server mic")
                appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, audioInput: null })
            }
        }
    }, [appState.audioContext, audioInputForGUI, appState.clientSetting.updateClientSetting])

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
                audioSrcNode.current = appState.audioContext!.createMediaElementSource(audio);
            }
            if (audioSrcNode.current.mediaElement != audio) {
                audioSrcNode.current = appState.audioContext!.createMediaElementSource(audio);
            }

            const dst = appState.audioContext.createMediaStreamDestination()
            audioSrcNode.current.connect(dst)
            appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, audioInput: dst.stream })

            const audio_echo = document.getElementById(AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK) as HTMLAudioElement
            audio_echo.srcObject = dst.stream
            audio_echo.play()
            audio_echo.volume = 0
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
    }, [audioInputForGUI, appState.clientSetting.updateClientSetting, fileInputEchoback])



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

    const audioOutputRecordingRow = useMemo(() => {
        // if (audioOutputForGUI != "record") {
        //     return <></>
        // }
        const onOutputRecordStartClicked = async () => {
            setOutputRecordingStarted(true)
            await appState.workletNodeSetting.startOutputRecording()
        }
        const onOutputRecordStopClicked = async () => {
            setOutputRecordingStarted(false)
            const record = await appState.workletNodeSetting.stopOutputRecording()
            downloadRecord(record)
        }

        const startClassName = outputRecordingStarted ? "body-button-active" : "body-button-stanby"
        const stopClassName = outputRecordingStarted ? "body-button-stanby" : "body-button-active"

        return (
            <div className="body-row split-3-3-4 left-padding-1  guided">
                <div className="body-item-title left-padding-2">output record</div>
                <div className="body-button-container">
                    <div onClick={onOutputRecordStartClicked} className={startClassName}>start</div>
                    <div onClick={onOutputRecordStopClicked} className={stopClassName}>stop</div>
                </div>
                <div className="body-input-container">
                </div>
            </div>
        )

    }, [audioOutputForGUI, outputRecordingStarted, appState.workletNodeSetting.startOutputRecording, appState.workletNodeSetting.stopOutputRecording])

    useEffect(() => {
        [AUDIO_ELEMENT_FOR_PLAY_RESULT, AUDIO_ELEMENT_FOR_TEST_ORIGINAL, AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK].forEach(x => {
            const audio = document.getElementById(x) as HTMLAudioElement
            if (audio) {
                if (audioOutputForGUI == "none") {
                    // @ts-ignore
                    audio.setSinkId("")
                    if (x == AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK) {
                        audio.volume = 0
                    } else {
                        audio.volume = 0
                    }
                } else {
                    // @ts-ignore
                    audio.setSinkId(audioOutputForGUI)
                    if (x == AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK) {
                        audio.volume = fileInputEchoback ? 1 : 0
                    } else {
                        audio.volume = 1
                    }
                }
            }
        })
    }, [audioOutputForGUI])


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
                {appState.frontendManagerState.stateControls.openDeviceSettingCheckbox.trigger}

                <div className="partition">
                    <div className="partition-header">
                        <span className="caret">
                            {accodionButton}
                        </span>
                        <span className="title" onClick={() => { appState.frontendManagerState.stateControls.openDeviceSettingCheckbox.updateState(!appState.frontendManagerState.stateControls.openDeviceSettingCheckbox.checked()) }}>
                            Device Setting
                        </span>
                        <span className="belongings">
                            <input className="belongings-checkbox" type="checkbox" checked={useServerMicrophone} onChange={(e) => {
                                setUseServerMicrophone(e.target.checked)
                            }} /> use server mic (Experimental)
                        </span>
                    </div>

                    <div className="partition-content">
                        {audioInputRow}
                        {audioMediaInputRow}
                        {audioOutputRow}
                        {audioOutputRecordingRow}
                    </div>
                </div>
            </>
        )
    }, [audioInputRow, audioMediaInputRow, audioOutputRow, audioOutputRecordingRow, useServerMicrophone])


    const downloadRecord = (data: Float32Array) => {

        const writeString = (view: DataView, offset: number, string: string) => {
            for (var i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        const floatTo16BitPCM = (output: DataView, offset: number, input: Float32Array) => {
            for (var i = 0; i < input.length; i++, offset += 2) {
                var s = Math.max(-1, Math.min(1, input[i]));
                output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            }
        };

        const buffer = new ArrayBuffer(44 + data.length * 2);
        const view = new DataView(buffer);

        // https://www.youfit.co.jp/archives/1418
        writeString(view, 0, 'RIFF');  // RIFFヘッダ
        view.setUint32(4, 32 + data.length * 2, true); // これ以降のファイルサイズ
        writeString(view, 8, 'WAVE'); // WAVEヘッダ
        writeString(view, 12, 'fmt '); // fmtチャンク
        view.setUint32(16, 16, true); // fmtチャンクのバイト数
        view.setUint16(20, 1, true); // フォーマットID
        view.setUint16(22, 1, true); // チャンネル数
        view.setUint32(24, 48000, true); // サンプリングレート
        view.setUint32(28, 48000 * 2, true); // データ速度
        view.setUint16(32, 2, true); // ブロックサイズ
        view.setUint16(34, 16, true); // サンプルあたりのビット数
        writeString(view, 36, 'data'); // dataチャンク
        view.setUint32(40, data.length * 2, true); // 波形データのバイト数
        floatTo16BitPCM(view, 44, data); // 波形データ
        const audioBlob = new Blob([view], { type: 'audio/wav' });

        const url = URL.createObjectURL(audioBlob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `output.wav`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    return {
        deviceSetting,
    }
}
