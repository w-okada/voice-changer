import { F0Detector } from "@dannadori/voice-changer-client-js"
import React, { useEffect, useMemo, useState } from "react"
import { useAppState } from "./001_provider/001_AppStateProvider";
import { AnimationTypes, HeaderButton, HeaderButtonProps } from "./components/101_HeaderButton";



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


export const useQualityControl = (): QualityControlState => {
    const appState = useAppState()
    const accodionButton = useMemo(() => {
        const accodionButtonProps: HeaderButtonProps = {
            stateControlCheckbox: appState.frontendManagerState.stateControls.openQualityControlCheckbox,
            tooltip: "Open/Close",
            onIcon: ["fas", "caret-up"],
            offIcon: ["fas", "caret-up"],
            animation: AnimationTypes.spinner,
            tooltipClass: "tooltip-right",
        };
        return <HeaderButton {...accodionButtonProps}></HeaderButton>;
    }, []);

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
                    <input type="checkbox" checked={appState.clientSetting.setting.echoCancel} onChange={(e) => {
                        appState.clientSetting.setEchoCancel(e.target.checked)
                    }} /> echo cancel
                </div>
                <div>
                    <input type="checkbox" checked={appState.clientSetting.setting.noiseSuppression} onChange={(e) => {
                        appState.clientSetting.setNoiseSuppression(e.target.checked)
                    }} /> suppression1
                </div>
                <div>
                    <input type="checkbox" checked={appState.clientSetting.setting.noiseSuppression2} onChange={(e) => {
                        appState.clientSetting.setNoiseSuppression2(e.target.checked)
                    }} /> suppression2
                </div>
                <div className="body-button-container">
                </div>
            </div>
        )
    }, [
        appState.clientSetting.setting.echoCancel, appState.clientSetting.setEchoCancel,
        appState.clientSetting.setting.noiseSuppression, appState.clientSetting.setNoiseSuppression,
        appState.clientSetting.setting.noiseSuppression2, appState.clientSetting.setNoiseSuppression2,
    ])

    const gainControlRow = useMemo(() => {
        return (
            <div className="body-row split-3-2-2-3 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Gain Control</div>
                <div>
                    <span className="body-item-input-slider-label">in</span>
                    <input type="range" className="body-item-input-slider" min="0.0" max="1.0" step="0.1" value={appState.clientSetting.setting.inputGain} onChange={(e) => {
                        appState.clientSetting.setInputGain(Number(e.target.value))
                    }}></input>
                    <span className="body-item-input-slider-val">{appState.clientSetting.setting.inputGain}</span>
                </div>
                <div>
                    <span className="body-item-input-slider-label">out</span>
                    <input type="range" className="body-item-input-slider" min="0.0" max="1.0" step="0.1" value={appState.clientSetting.setting.outputGain} onChange={(e) => {
                        appState.clientSetting.setOutputGain(Number(e.target.value))
                    }}></input>
                    <span className="body-item-input-slider-val">{appState.clientSetting.setting.outputGain}</span>
                </div>
                <div className="body-button-container">
                </div>
            </div>
        )
    }, [
        appState.clientSetting.setting.inputGain, appState.clientSetting.setting.inputGain,
        appState.clientSetting.setting.outputGain, appState.clientSetting.setOutputGain,
    ])

    const f0DetectorRow = useMemo(() => {
        const desc = { "harvest": "High Quality", "dio": "Light Weight" }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">F0 Detector</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.serverSetting.setting.f0Detector} onChange={(e) => {
                        appState.serverSetting.setF0Detector(e.target.value as F0Detector)
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
    }, [appState.serverSetting.setting.f0Detector, appState.serverSetting.setF0Detector])


    const recordIORow = useMemo(() => {
        const setReocrdIO = async (val: number) => {
            await appState.serverSetting.setRecordIO(val)
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
                        <select className="body-select" value={appState.serverSetting.setting.recordIO} onChange={(e) => {
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
    }, [appState.serverSetting.setting.recordIO, appState.serverSetting.setRecordIO, outputAudioDeviceInfo, audioOutputForGUI])

    const QualityControlContent = useMemo(() => {
        return (
            <>
                {noiseControlRow}
                {gainControlRow}
                {f0DetectorRow}
                {recordIORow}
            </>
        )
    }, [gainControlRow, noiseControlRow, f0DetectorRow, recordIORow])


    const qualityControl = useMemo(() => {
        return (
            <>
                {appState.frontendManagerState.stateControls.openQualityControlCheckbox.trigger}
                <div className="partition">
                    <div className="partition-header">
                        <span className="caret">
                            {accodionButton}
                        </span>
                        <span className="title" onClick={() => { appState.frontendManagerState.stateControls.openQualityControlCheckbox.updateState(!appState.frontendManagerState.stateControls.openQualityControlCheckbox.checked()) }}>
                            Quality Control
                        </span>
                    </div>

                    <div className="partition-content">
                        {QualityControlContent}
                    </div>
                </div>
            </>
        )
    }, [QualityControlContent])

    return {
        qualityControl,
    }

}


