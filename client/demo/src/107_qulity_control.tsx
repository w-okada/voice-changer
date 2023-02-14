import { BufferSize, DownSamplingMode, F0Detector, Protocol, SampleRate, VoiceChangerMode } from "@dannadori/voice-changer-client-js"
import React, { useEffect, useMemo, useState } from "react"
import { ClientState } from "@dannadori/voice-changer-client-js";


export type UseQualityControlProps = {
    clientState: ClientState
}

export type QualityControlState = {
    qualityControl: JSX.Element;
}
const reloadDevices = async () => {
    try {
        const ms = await navigator.mediaDevices.getUserMedia({ video: false, audio: true });
        ms.getTracks().forEach(x => { x.stop() })
    } catch (e) {
        console.warn("Enumerate device error::", e)
    }
    const mediaDeviceInfos = await navigator.mediaDevices.enumerateDevices();
    const audioOutputs = mediaDeviceInfos.filter(x => { return x.kind == "audiooutput" })

    return audioOutputs
}


export const useQualityControl = (props: UseQualityControlProps): QualityControlState => {
    const [showQualityControl, setShowQualityControl] = useState<boolean>(false)
    const [outputAudioDeviceInfo, setOutputAudioDeviceInfo] = useState<MediaDeviceInfo[]>([])
    const [audioOutputForGUI, setAudioOutputForGUI] = useState<string>("none")
    useEffect(() => {
        const initialize = async () => {
            const audioInfo = await reloadDevices()
            setOutputAudioDeviceInfo(audioInfo)
        }
        initialize()
    }, [])


    const noiseControlRow = useMemo(() => {
        return (
            <div className="body-row split-3-2-2-2-1 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Noise Suppression</div>
                <div>
                    <input type="checkbox" checked={props.clientState.clientSetting.setting.echoCancel} onChange={(e) => {
                        props.clientState.clientSetting.setEchoCancel(e.target.checked)
                    }} /> echo cancel
                </div>
                <div>
                    <input type="checkbox" checked={props.clientState.clientSetting.setting.noiseSuppression} onChange={(e) => {
                        props.clientState.clientSetting.setNoiseSuppression(e.target.checked)
                    }} /> suppression1
                </div>
                <div>
                    <input type="checkbox" checked={props.clientState.clientSetting.setting.noiseSuppression2} onChange={(e) => {
                        props.clientState.clientSetting.setNoiseSuppression2(e.target.checked)
                    }} /> suppression2
                </div>
                <div className="body-button-container">
                </div>
            </div>
        )
    }, [
        props.clientState.clientSetting.setting.echoCancel, props.clientState.clientSetting.setEchoCancel,
        props.clientState.clientSetting.setting.noiseSuppression, props.clientState.clientSetting.setNoiseSuppression,
        props.clientState.clientSetting.setting.noiseSuppression2, props.clientState.clientSetting.setNoiseSuppression2,
    ])

    const gainControlRow = useMemo(() => {
        return (
            <div className="body-row split-3-2-2-3 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Gain Control</div>
                <div>
                    <span className="body-item-input-slider-label">in</span>
                    <input type="range" className="body-item-input-slider" min="0.0" max="1.0" step="0.1" value={props.clientState.clientSetting.setting.inputGain} onChange={(e) => {
                        props.clientState.clientSetting.setInputGain(Number(e.target.value))
                    }}></input>
                    <span className="body-item-input-slider-val">{props.clientState.clientSetting.setting.inputGain}</span>
                </div>
                <div>
                    <span className="body-item-input-slider-label">out</span>
                    <input type="range" className="body-item-input-slider" min="0.0" max="1.0" step="0.1" value={props.clientState.clientSetting.setting.outputGain} onChange={(e) => {
                        props.clientState.clientSetting.setOutputGain(Number(e.target.value))
                    }}></input>
                    <span className="body-item-input-slider-val">{props.clientState.clientSetting.setting.outputGain}</span>
                </div>
                <div className="body-button-container">
                </div>
            </div>
        )
    }, [
        props.clientState.clientSetting.setting.inputGain, props.clientState.clientSetting.setting.inputGain,
        props.clientState.clientSetting.setting.outputGain, props.clientState.clientSetting.setOutputGain,
    ])

    const f0DetectorRow = useMemo(() => {
        const desc = { "harvest": "High Quality", "dio": "Light Weight" }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">F0 Detector</div>
                <div className="body-select-container">
                    <select className="body-select" value={props.clientState.serverSetting.setting.f0Detector} onChange={(e) => {
                        props.clientState.serverSetting.setF0Detector(e.target.value as F0Detector)
                    }}>
                        {
                            Object.values(F0Detector).map(x => {
                                //@ts-ignore
                                return <option key={x} value={x}>{x}({desc[x]})</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [props.clientState.serverSetting.setting.f0Detector, props.clientState.serverSetting.setF0Detector])


    const recordIORow = useMemo(() => {
        const setReocrdIO = async (val: number) => {
            await props.clientState.serverSetting.setRecordIO(val)
            if (val == 0) {
                const imageContainer = document.getElementById("quality-control-analyze-image-container") as HTMLDivElement
                imageContainer.innerHTML = ""
                const image = document.createElement("img")
                image.src = "/tmp/analyze-dio.png?" + new Date().getTime()
                imageContainer.appendChild(image)
                const image2 = document.createElement("img")
                image2.src = "/tmp/analyze-harvest.png?" + new Date().getTime()
                imageContainer.appendChild(image2)

                const wavContainer = document.getElementById("quality-control-analyze-wav-container") as HTMLDivElement
                wavContainer.innerHTML = ""
                const media1 = document.createElement("audio") as HTMLAudioElement
                media1.src = "/tmp/in.wav?" + new Date().getTime()
                media1.controls = true
                // @ts-ignore
                media1.setSinkId(audioOutputForGUI)
                wavContainer.appendChild(media1)
                const media2 = document.createElement("audio") as HTMLAudioElement
                media2.src = "/tmp/out.wav?" + new Date().getTime()
                media2.controls = true
                // @ts-ignore
                media2.setSinkId(audioOutputForGUI)
                wavContainer.appendChild(media2)
            }
        }
        return (
            <>
                <div className="body-row split-3-7 left-padding-1 guided">
                    <div className="body-item-title left-padding-1 ">recordIO</div>
                    <div className="body-select-container">
                        <select className="body-select" value={props.clientState.serverSetting.setting.recordIO} onChange={(e) => {
                            setReocrdIO(Number(e.target.value))
                        }}>
                            {
                                Object.values([0, 1]).map(x => {
                                    return <option key={x} value={x}>{x}</option>
                                })
                            }
                        </select>
                    </div>
                </div>
                <div className="body-row split-3-7 left-padding-1 guided">
                    <div className="body-item-title left-padding-1 ">
                        <div>
                            Spectrogram
                        </div>
                        <div>
                            <span>(left: dio, right:harvest)</span>
                        </div>
                    </div>
                    <div className="body-image-container-quality-analyze" id="quality-control-analyze-image-container">
                    </div>
                </div>
                <div className="body-row split-3-7 left-padding-1 guided">
                    <div className="body-item-title left-padding-1 ">
                        <div>
                            wav (left:input, right:output)
                        </div>
                        <select className="body-select" value={audioOutputForGUI} onChange={(e) => {
                            setAudioOutputForGUI(e.target.value)
                            const wavContainer = document.getElementById("quality-control-analyze-wav-container") as HTMLDivElement
                            wavContainer.childNodes.forEach(x => {
                                if (x instanceof HTMLAudioElement) {
                                    //@ts-ignore
                                    x.setSinkId(e.target.value)
                                }
                            })
                        }}>
                            {
                                outputAudioDeviceInfo.map(x => {
                                    return <option key={x.deviceId} value={x.deviceId}>{x.label}</option>
                                })
                            }
                        </select>
                    </div>
                    <div className="body-wav-container-quality-analyze" id="quality-control-analyze-wav-container">
                    </div>
                </div>
            </>
        )
    }, [props.clientState.serverSetting.setting.recordIO, props.clientState.serverSetting.setRecordIO, outputAudioDeviceInfo, audioOutputForGUI])

    const QualityControlContent = useMemo(() => {
        if (!showQualityControl) return <></>
        return (
            <>
                {noiseControlRow}
                {gainControlRow}
                {f0DetectorRow}
                {recordIORow}
            </>
        )
    }, [showQualityControl, gainControlRow, noiseControlRow, f0DetectorRow, recordIORow,])


    const qualityControl = useMemo(() => {
        return (
            <>
                <div className="body-row split-3-7 left-padding-1">
                    <div className="body-sub-section-title">Quality Control</div>
                    <div>
                        <input type="checkbox" checked={showQualityControl} onChange={(e) => {
                            setShowQualityControl(e.target.checked)
                        }} /> show
                    </div>
                </div>
                {QualityControlContent}
            </>
        )
    }, [showQualityControl, QualityControlContent])

    return {
        qualityControl,
    }

}


