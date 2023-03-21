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

    const [recording, setRecording] = useState<boolean>(false)
    const [outputAudioDeviceInfo, setOutputAudioDeviceInfo] = useState<MediaDeviceInfo[]>([])
    const [audioOutputForGUI, setAudioOutputForGUI] = useState<string>("default")
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
                    <input type="checkbox" checked={appState.clientSetting.clientSetting.echoCancel} onChange={(e) => {
                        appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, echoCancel: e.target.checked })
                    }} /> echo cancel
                </div>
                <div>
                    <input type="checkbox" checked={appState.clientSetting.clientSetting.noiseSuppression} onChange={(e) => {
                        appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, noiseSuppression: e.target.checked })
                    }} /> suppression1
                </div>
                <div>
                    <input type="checkbox" checked={appState.clientSetting.clientSetting.noiseSuppression2} onChange={(e) => {
                        appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, noiseSuppression2: e.target.checked })
                    }} /> suppression2
                </div>
                <div className="body-button-container">
                </div>
            </div>
        )
    }, [
        appState.clientSetting.clientSetting.echoCancel,
        appState.clientSetting.clientSetting.noiseSuppression,
        appState.clientSetting.clientSetting.noiseSuppression2,
        appState.clientSetting.updateClientSetting
    ])

    const gainControlRow = useMemo(() => {
        return (
            <div className="body-row split-3-2-2-3 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Gain Control</div>
                <div>
                    <span className="body-item-input-slider-label">in</span>
                    <input type="range" className="body-item-input-slider" min="0.0" max="10.0" step="0.1" value={appState.clientSetting.clientSetting.inputGain} onChange={(e) => {
                        appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, inputGain: Number(e.target.value) })
                    }}></input>
                    <span className="body-item-input-slider-val">{appState.clientSetting.clientSetting.inputGain}</span>
                </div>
                <div>
                    <span className="body-item-input-slider-label">out</span>
                    <input type="range" className="body-item-input-slider" min="0.0" max="10.0" step="0.1" value={appState.clientSetting.clientSetting.outputGain} onChange={(e) => {
                        appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, outputGain: Number(e.target.value) })
                    }}></input>
                    <span className="body-item-input-slider-val">{appState.clientSetting.clientSetting.outputGain}</span>
                </div>
                <div className="body-button-container">
                </div>
            </div>
        )
    }, [
        appState.clientSetting.clientSetting.inputGain,
        appState.clientSetting.clientSetting.outputGain,
        appState.clientSetting.updateClientSetting
    ])

    // const f0DetectorRow = useMemo(() => {
    //     const desc = { "harvest": "High Quality", "dio": "Light Weight" }
    //     return (
    //         <div className="body-row split-3-7 left-padding-1 guided">
    //             <div className="body-item-title left-padding-1 ">F0 Detector</div>
    //             <div className="body-select-container">
    //                 <select className="body-select" value={appState.serverSetting.serverSetting.f0Detector} onChange={(e) => {
    //                     appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, f0Detector: e.target.value as F0Detector })
    //                 }}>
    //                     {
    //                         Object.values(F0Detector).map(x => {
    //                             //@ts-ignore
    //                             return <option key={x} value={x}>{x}({desc[x]})</option>
    //                         })
    //                     }
    //                 </select>
    //             </div>
    //         </div>
    //     )
    // }, [appState.serverSetting.serverSetting.f0Detector, appState.serverSetting.updateServerSettings])


    const recordIORow = useMemo(() => {
        const onRecordStartClicked = async () => {
            setRecording(true)
            await appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, recordIO: 1 })
        }
        const onRecordStopClicked = async () => {
            setRecording(false)
            await appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, recordIO: 0 })

            // set wav (input)
            const wavInput = document.getElementById("body-wav-container-wav-input") as HTMLAudioElement
            wavInput.src = "/tmp/in.wav?" + new Date().getTime()
            wavInput.controls = true
            // @ts-ignore
            wavInput.setSinkId(audioOutputForGUI)

            // set wav (output)
            const wavOutput = document.getElementById("body-wav-container-wav-output") as HTMLAudioElement
            wavOutput.src = "/tmp/out.wav?" + new Date().getTime()
            wavOutput.controls = true
            // @ts-ignore
            wavOutput.setSinkId(audioOutputForGUI)
        }
        const onRecordAnalizeClicked = async () => {
            if (appState.frontendManagerState.isConverting) {
                alert("please stop voice conversion. 解析処理と音声変換を同時に行うことはできません。音声変化をストップしてください。")
                return
            }
            appState.frontendManagerState.setIsAnalyzing(true)
            await appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, recordIO: 2 })
            // set spectrogram (dio)
            const imageDio = document.getElementById("body-image-container-img-dio") as HTMLImageElement
            imageDio.src = "/tmp/analyze-dio.png?" + new Date().getTime()
            imageDio.style.width = "100%"

            // set spectrogram (harvest)
            const imageHarvest = document.getElementById("body-image-container-img-harvest") as HTMLImageElement
            imageHarvest.src = "/tmp/analyze-harvest.png?" + new Date().getTime()
            imageHarvest.style.width = "100%"

            appState.frontendManagerState.setIsAnalyzing(false)
        }

        const startClassName = recording ? "body-button-active" : "body-button-stanby"
        const stopClassName = recording ? "body-button-stanby" : "body-button-active"
        const analyzeClassName = appState.frontendManagerState.isAnalyzing ? "body-button-active" : "body-button-stanby"
        const analyzeLabel = appState.frontendManagerState.isAnalyzing ? "wait..." : "Analyze"

        return (
            <>
                <div className="body-row split-3-7 left-padding-1 guided">
                    <div className="body-item-title left-padding-1 ">Analyzer(Experimental)</div>
                    <div className="body-button-container">
                    </div>
                </div>
                <div className="body-row split-3-7 left-padding-1 guided">
                    <div className="body-item-title left-padding-2 ">
                        Sampling
                    </div>
                    <div className="body-button-container">
                        <div onClick={onRecordStartClicked} className={startClassName}>Start</div>
                        <div onClick={onRecordStopClicked} className={stopClassName}>Stop</div>
                        {/* <div onClick={onRecordAnalizeClicked} className={analyzeClassName}>{analyzeLabel}</div> */}
                    </div>
                </div>

                <div className="body-row split-3-2-2-3 left-padding-1 guided">
                    <div className="body-item-title left-padding-2 ">
                        <div>
                            Play
                        </div>
                        <select className="body-select-50 left-margin-2" value={audioOutputForGUI} onChange={(e) => {
                            setAudioOutputForGUI(e.target.value)
                            const wavInput = document.getElementById("body-wav-container-wav-input") as HTMLAudioElement
                            const wavOutput = document.getElementById("body-wav-container-wav-output") as HTMLAudioElement
                            //@ts-ignore
                            wavInput.setSinkId(e.target.value)
                            //@ts-ignore
                            wavOutput.setSinkId(e.target.value)
                        }}>
                            {
                                outputAudioDeviceInfo.map(x => {
                                    return <option key={x.deviceId} value={x.deviceId}>{x.label}</option>
                                })
                            }
                        </select>
                    </div>
                    {/* <div>
                        <div className="body-wav-container">
                            <div className="body-wav-container-title">Input</div>
                            <div className="body-wav-container-title">Output</div>
                        </div>
                        <div className="body-wav-container">
                            <div className="body-wav-container-wav">
                                <audio src="" id="body-wav-container-wav-input"></audio>
                            </div>
                            <div className="body-wav-container-wav" >
                                <audio src="" id="body-wav-container-wav-output"></audio>
                            </div>
                        </div>
                    </div> */}
                    <div>
                        <div className="body-wav-container-title">Input</div>
                        <div className="body-wav-container-wav">
                            <audio src="" id="body-wav-container-wav-input"></audio>
                        </div>
                    </div>
                    <div >
                        <div className="body-wav-container-title">Output</div>
                        <div className="body-wav-container-wav" >
                            <audio src="" id="body-wav-container-wav-output"></audio>
                        </div>
                    </div>
                    <div></div>
                </div>
                {/* 
                <div className="body-row split-3-7 left-padding-1 guided">
                    <div className="body-item-title left-padding-2 ">
                        Spectrogram
                    </div>
                    <div>
                        <div className="body-image-container">
                            <div className="body-image-container-title">PyWorld Dio</div>
                            <div className="body-image-container-title">PyWorld Harvest</div>
                        </div>
                        <div className="body-image-container">
                            <div className="body-image-container-img" >
                                <img src="" alt="" id="body-image-container-img-dio" />
                            </div>
                            <div className="body-image-container-img">
                                <img src="" alt="" id="body-image-container-img-harvest" />

                            </div>
                        </div>
                    </div>
                </div> */}

            </>
        )
    }, [appState.serverSetting.serverSetting.recordIO, appState.serverSetting.updateServerSettings, outputAudioDeviceInfo, audioOutputForGUI, appState.frontendManagerState.isAnalyzing, appState.frontendManagerState.isConverting])

    const QualityControlContent = useMemo(() => {
        return (
            <>
                {noiseControlRow}
                {gainControlRow}
                <div className="body-row divider"></div>
                {recordIORow}
            </>
        )
    }, [gainControlRow, noiseControlRow, recordIORow])


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


