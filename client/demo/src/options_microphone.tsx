import * as React from "react";
import { useEffect, useMemo, useState } from "react";
import { CHROME_EXTENSION } from "./const";
import { DefaultVoiceChangerRequestParamas, VoiceChangerOptions, VoiceChangerRequestParamas, DefaultVoiceChangerOptions, SampleRate, BufferSize, VoiceChangerMode, Protocol } from "@dannadori/voice-changer-client-js"


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
    return audioInputs
}


export type MicrophoneOptionsComponent = {
    component: JSX.Element,
    options: VoiceChangerOptions,
    params: VoiceChangerRequestParamas
    isStarted: boolean
}

export const useMicrophoneOptions = (): MicrophoneOptionsComponent => {
    // GUI Info
    const [audioDeviceInfo, setAudioDeviceInfo] = useState<MediaDeviceInfo[]>([])
    const [editSpeakerTargetId, setEditSpeakerTargetId] = useState<number>(0)
    const [editSpeakerTargetName, setEditSpeakerTargetName] = useState<string>("")

    // const [options, setOptions] = useState<MicrophoneOptionsState>(InitMicrophoneOptionsState)
    const [params, setParams] = useState<VoiceChangerRequestParamas>(DefaultVoiceChangerRequestParamas)
    const [options, setOptions] = useState<VoiceChangerOptions>(DefaultVoiceChangerOptions)
    const [isStarted, setIsStarted] = useState<boolean>(false)

    useEffect(() => {
        const initialize = async () => {
            const audioInfo = await reloadDevices()
            setAudioDeviceInfo(audioInfo)

            if (CHROME_EXTENSION) {
                //@ts-ignore
                const storedOptions = await chrome.storage.local.get("microphoneOptions")
                if (storedOptions) {
                    setOptions(storedOptions)
                }
            }
        }
        initialize()
    }, [])

    useEffect(() => {
        const storeOptions = async () => {
            if (CHROME_EXTENSION) {
                // @ts-ignore
                await chrome.storage.local.set({ microphoneOptions: options })
            }
        }
        storeOptions()
    }, [options]) // loadより前に持ってくるとstorage内が初期化されるのでだめかも。（要検証）


    const startButtonRow = useMemo(() => {
        const onStartClicked = () => {
            setIsStarted(true)
        }
        const onStopClicked = () => {
            setIsStarted(false)
        }
        const startClassName = isStarted ? "body-button-active" : "body-button-stanby"
        const stopClassName = isStarted ? "body-button-stanby" : "body-button-active"
        console.log("ClassName", startClassName, stopClassName)

        return (
            <div className="body-row split-3-3-4 left-padding-1">
                <div className="body-item-title">Start</div>
                <div className="body-button-container">
                    <div onClick={onStartClicked} className={startClassName}>start</div>
                    <div onClick={onStopClicked} className={stopClassName}>stop</div>
                </div>
                <div className="body-input-container">
                </div>
            </div>

        )
    }, [isStarted])


    const setAudioInputDeviceId = async (deviceId: string) => {
        setOptions({ ...options, audioInputDeviceId: deviceId })
    }

    const onSetServerClicked = async () => {
        const input = document.getElementById("mmvc-server-url") as HTMLInputElement
        setOptions({ ...options, mmvcServerUrl: input.value })
    }
    const onProtocolChanged = async (val: Protocol) => {
        setOptions({ ...options, protocol: val })
    }

    const onSampleRateChanged = async (val: SampleRate) => {
        setOptions({ ...options, sampleRate: val })
    }
    const onBufferSizeChanged = async (val: BufferSize) => {
        setOptions({ ...options, bufferSize: val })
    }
    const onChunkSizeChanged = async (val: number) => {
        setOptions({ ...options, inputChunkNum: val })
    }
    const onSrcIdChanged = async (val: number) => {
        setParams({ ...params, srcId: val })
    }
    const onDstIdChanged = async (val: number) => {
        setParams({ ...params, dstId: val })
    }
    const onSetSpeakerMappingClicked = async () => {
        const targetId = editSpeakerTargetId
        const targetName = editSpeakerTargetName
        const targetSpeaker = options.speakers.find(x => { return x.id == targetId })
        if (targetSpeaker) {
            if (targetName.length == 0) { // Delete
                const newSpeakers = options.speakers.filter(x => { return x.id != targetId })
                options.speakers = newSpeakers
            } else { // Update
                targetSpeaker.name = targetName
            }
        } else {
            if (targetName.length == 0) { // Noop
            } else {// add
                options.speakers.push({
                    id: targetId,
                    name: targetName
                })
            }
        }
        setOptions({ ...options })
    }

    const onVfEnabledChange = async (val: boolean) => {
        setOptions({ ...options, forceVfDisable: val })
    }
    const onVoiceChangeModeChanged = async (val: VoiceChangerMode) => {
        setOptions({ ...options, voiceChangerMode: val })
    }
    const onGpuChanged = async (val: number) => {
        setParams({ ...params, gpu: val })
    }
    const onCrossFadeLowerValueChanged = async (val: number) => {
        setParams({ ...params, crossFadeLowerValue: val })
    }
    const onCrossFadeOffsetRateChanged = async (val: number) => {
        setParams({ ...params, crossFadeOffsetRate: val })
    }
    const onCrossFadeEndRateChanged = async (val: number) => {
        setParams({ ...params, crossFadeEndRate: val })
    }

    const settings = useMemo(() => {
        return (
            <>
                <div className="body-row left-padding-1">
                    <div className="body-section-title">Virtual Microphone</div>
                </div>
                {startButtonRow}

                <div className="body-row split-3-3-4 left-padding-1">
                    <div className="body-item-title">MMVC Server</div>
                    <div className="body-input-container">
                        <input type="text" defaultValue={options.mmvcServerUrl} id="mmvc-server-url" className="body-item-input" />
                    </div>
                    <div className="body-button-container">
                        <div className="body-button" onClick={onSetServerClicked}>set</div>
                    </div>
                </div>

                <div className="body-row split-3-7 left-padding-1 highlight">
                    <div className="body-item-title">Protocol</div>
                    <div className="body-select-container">
                        <select className="body-select" value={options.protocol} onChange={(e) => {
                            onProtocolChanged(e.target.value as
                                Protocol)
                        }}>
                            {
                                Object.values(Protocol).map(x => {
                                    return <option key={x} value={x}>{x}</option>
                                })
                            }
                        </select>
                    </div>
                </div>

                <div className="body-row split-3-7 left-padding-1 highlight">
                    <div className="body-item-title">Microphone</div>
                    <div className="body-select-container">
                        <select className="body-select" value={options.audioInputDeviceId || "none"} onChange={(e) => { setAudioInputDeviceId(e.target.value) }}>
                            {
                                audioDeviceInfo.map(x => {
                                    return <option key={x.deviceId} value={x.deviceId}>{x.label}</option>
                                })
                            }
                        </select>
                    </div>
                </div>

                <div className="body-row split-3-7 left-padding-1 highlight">
                    <div className="body-item-title">Sample Rate</div>
                    <div className="body-select-container">
                        <select className="body-select" value={options.sampleRate} onChange={(e) => { onSampleRateChanged(Number(e.target.value) as SampleRate) }}>
                            {
                                Object.values(SampleRate).map(x => {
                                    return <option key={x} value={x}>{x}</option>
                                })
                            }
                        </select>
                    </div>
                </div>

                <div className="body-row split-3-7 left-padding-1 highlight">
                    <div className="body-item-title">Buffer Size</div>
                    <div className="body-select-container">
                        <select className="body-select" value={options.bufferSize} onChange={(e) => { onBufferSizeChanged(Number(e.target.value) as BufferSize) }}>
                            {
                                Object.values(BufferSize).map(x => {
                                    return <option key={x} value={x}>{x}</option>
                                })
                            }
                        </select>
                    </div>
                </div>

                <div className="body-row split-3-7 left-padding-1 highlight">
                    <div className="body-item-title">Chunk Num(128sample/chunk)</div>
                    <div className="body-input-container">
                        <input type="number" min={1} max={256} step={1} value={options.inputChunkNum} onChange={(e) => { onChunkSizeChanged(Number(e.target.value)) }} />
                    </div>
                </div>

                <div className="body-row split-3-3-4 left-padding-1 highlight">
                    <div className="body-item-title">VF Enabled</div>
                    <div>
                        <input type="checkbox" checked={options.forceVfDisable} onChange={(e) => onVfEnabledChange(e.target.checked)} />
                    </div>
                    <div className="body-button-container">
                    </div>
                </div>

                <div className="body-row split-3-7 left-padding-1 highlight">
                    <div className="body-item-title">Voice Change Mode</div>
                    <div className="body-select-container">
                        <select className="body-select" value={options.voiceChangerMode} onChange={(e) => { onVoiceChangeModeChanged(e.target.value as VoiceChangerMode) }}>
                            {
                                Object.values(VoiceChangerMode).map(x => {
                                    return <option key={x} value={x}>{x}</option>
                                })
                            }
                        </select>
                    </div>
                </div>

                <div className="body-row split-3-7 left-padding-1 highlight">
                    <div className="body-item-title">Source Speaker Id</div>
                    <div className="body-select-container">
                        <select className="body-select" value={params.srcId} onChange={(e) => { onSrcIdChanged(Number(e.target.value)) }}>
                            {
                                options.speakers.map(x => {
                                    return <option key={x.id} value={x.id}>{x.name}({x.id})</option>
                                })
                            }
                        </select>
                    </div>
                </div>

                <div className="body-row split-3-7 left-padding-1 highlight">
                    <div className="body-item-title">Destination Speaker Id</div>
                    <div className="body-select-container">
                        <select className="body-select" value={params.dstId} onChange={(e) => { onDstIdChanged(Number(e.target.value)) }}>
                            {
                                options.speakers.map(x => {
                                    return <option key={x.id} value={x.id}>{x.name}({x.id})</option>
                                })
                            }
                        </select>
                    </div>
                </div>

                <div className="body-row split-3-1-2-4 left-padding-1 highlight">
                    <div className="body-item-title">Edit Speaker Mapping</div>
                    <div className="body-input-container">
                        <input type="number" min={1} max={256} step={1} value={editSpeakerTargetId} onChange={(e) => {
                            const id = Number(e.target.value)
                            setEditSpeakerTargetId(id)
                            setEditSpeakerTargetName(options.speakers.find(x => { return x.id == id })?.name || "")
                        }} />
                    </div>
                    <div className="body-input-container">
                        <input type="text" value={editSpeakerTargetName} onChange={(e) => { setEditSpeakerTargetName(e.target.value) }} />
                    </div>
                    <div className="body-button-container">
                        <div className="body-button" onClick={onSetSpeakerMappingClicked}>set</div>
                    </div>
                </div>

                <div className="body-row split-3-7 left-padding-1 highlight">
                    <div className="body-item-title">GPU</div>
                    <div className="body-input-container">
                        <input type="number" min={-1} max={5} step={1} value={params.gpu} onChange={(e) => { onGpuChanged(Number(e.target.value)) }} />
                    </div>
                </div>


                <div className="body-row split-3-7 left-padding-1 highlight">
                    <div className="body-item-title">Cross Fade Lower Val</div>
                    <div className="body-input-container">
                        <input type="number" min={0} max={1} step={0.1} value={params.crossFadeLowerValue} onChange={(e) => { onCrossFadeLowerValueChanged(Number(e.target.value)) }} />
                    </div>
                </div>

                <div className="body-row split-3-7 left-padding-1 highlight">
                    <div className="body-item-title">Cross Fade Offset Rate</div>
                    <div className="body-input-container">
                        <input type="number" min={0} max={1} step={0.1} value={params.crossFadeOffsetRate} onChange={(e) => { onCrossFadeOffsetRateChanged(Number(e.target.value)) }} />
                    </div>
                </div>

                <div className="body-row split-3-7 left-padding-1 highlight">
                    <div className="body-item-title">Cross Fade End Rate</div>
                    <div className="body-input-container">
                        <input type="number" min={0} max={1} step={0.1} value={params.crossFadeEndRate} onChange={(e) => { onCrossFadeEndRateChanged(Number(e.target.value)) }} />
                    </div>
                </div>
            </>
        )
    }, [audioDeviceInfo, editSpeakerTargetId, editSpeakerTargetName, startButtonRow, params, options])

    return {
        component: settings,
        params: params,
        options: options,
        isStarted
    }
}

