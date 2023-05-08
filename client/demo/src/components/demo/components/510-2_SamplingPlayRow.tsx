import React, { useMemo } from "react"
import { AUDIO_ELEMENT_FOR_SAMPLING_INPUT, AUDIO_ELEMENT_FOR_SAMPLING_OUTPUT } from "../../../const"
import { useGuiState } from "./../001_GuiStateProvider"

export const SamplingPlayRow = () => {
    const guiState = useGuiState()

    const samplingPlayRow = useMemo(() => {
        return (
            <div className="body-row split-3-2-2-3 left-padding-1 guided">
                <div className="body-item-title left-padding-2 ">
                    <div>
                        Play
                    </div>
                    <select className="body-select-50 left-margin-2" value={guiState.audioOutputForAnalyzer} onChange={(e) => {
                        guiState.setAudioOutputForAnalyzer(e.target.value)
                        const wavInput = document.getElementById(AUDIO_ELEMENT_FOR_SAMPLING_INPUT) as HTMLAudioElement
                        const wavOutput = document.getElementById(AUDIO_ELEMENT_FOR_SAMPLING_OUTPUT) as HTMLAudioElement
                        //@ts-ignore
                        wavInput.setSinkId(e.target.value)
                        //@ts-ignore
                        wavOutput.setSinkId(e.target.value)
                    }}>
                        {
                            guiState.outputAudioDeviceInfo.map(x => {
                                if (x.deviceId == "none") {
                                    return null
                                }
                                return <option key={x.deviceId} value={x.deviceId}>{x.label}</option>
                            }).filter(x => { return x != null })
                        }
                    </select>
                </div>
                <div>
                    <div className="body-wav-container-title">Input</div>
                    <div className="body-wav-container-wav">
                        <audio src="" id={AUDIO_ELEMENT_FOR_SAMPLING_INPUT}></audio>
                    </div>
                </div>
                <div >
                    <div className="body-wav-container-title">Output</div>
                    <div className="body-wav-container-wav" >
                        <audio src="" id={AUDIO_ELEMENT_FOR_SAMPLING_OUTPUT}></audio>
                    </div>
                </div>
                <div></div>
            </div>
        )
    }, [guiState.audioOutputForAnalyzer, guiState.outputAudioDeviceInfo])

    return samplingPlayRow
}