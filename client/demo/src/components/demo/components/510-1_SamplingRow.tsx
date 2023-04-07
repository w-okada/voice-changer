import React, { useMemo, useState } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { AUDIO_ELEMENT_FOR_SAMPLING_INPUT, AUDIO_ELEMENT_FOR_SAMPLING_OUTPUT } from "../../../const"
import { useGuiState } from "./../001_GuiStateProvider"

export const SamplingRow = () => {
    const [recording, setRecording] = useState<boolean>(false)
    const appState = useAppState()
    const guiState = useGuiState()


    const samplingRow = useMemo(() => {

        const onRecordStartClicked = async () => {
            setRecording(true)
            await appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, recordIO: 1 })
        }
        const onRecordStopClicked = async () => {
            setRecording(false)
            await appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, recordIO: 0 })

            // set wav (input)
            const wavInput = document.getElementById(AUDIO_ELEMENT_FOR_SAMPLING_INPUT) as HTMLAudioElement
            wavInput.src = "/tmp/in.wav?" + new Date().getTime()
            wavInput.controls = true
            // @ts-ignore
            wavInput.setSinkId(guiState.audioOutputForAnalyzer)

            // set wav (output)
            const wavOutput = document.getElementById(AUDIO_ELEMENT_FOR_SAMPLING_OUTPUT) as HTMLAudioElement
            wavOutput.src = "/tmp/out.wav?" + new Date().getTime()
            wavOutput.controls = true
            // @ts-ignore
            wavOutput.setSinkId(guiState.audioOutputForAnalyzer)
        }

        const startClassName = recording ? "body-button-active" : "body-button-stanby"
        const stopClassName = recording ? "body-button-stanby" : "body-button-active"
        return (
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
        )
    }, [recording, appState.serverSetting.serverSetting, appState.serverSetting.updateServerSettings, guiState.audioOutputForAnalyzer])

    return samplingRow
}