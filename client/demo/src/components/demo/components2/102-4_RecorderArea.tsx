import React, { useMemo, useState } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"
import { AUDIO_ELEMENT_FOR_SAMPLING_INPUT, AUDIO_ELEMENT_FOR_SAMPLING_OUTPUT } from "../../../const"

export type RecorderAreaProps = {
}

export const RecorderArea = (_props: RecorderAreaProps) => {
    const { serverSetting, workletNodeSetting } = useAppState()
    const { audioOutputForAnalyzer, setAudioOutputForAnalyzer, outputAudioDeviceInfo } = useGuiState()

    const [serverIORecording, setServerIORecording] = useState<boolean>(false)

    // const recorderRow = useMemo(() => {
    //     return (
    //         <div className="config-sub-area-control">
    //             <div className="config-sub-area-control-title">RECORD:</div>
    //             <div className="config-sub-area-control-field">
    //             </div>
    //         </div>
    //     )
    // }, [serverSetting.serverSetting, serverSetting.updateServerSettings])



    const serverIORecorderRow = useMemo(() => {
        const onServerIORecordStartClicked = async () => {
            setServerIORecording(true)
            await serverSetting.updateServerSettings({ ...serverSetting.serverSetting, recordIO: 1 })
        }
        const onServerIORecordStopClicked = async () => {
            setServerIORecording(false)
            await serverSetting.updateServerSettings({ ...serverSetting.serverSetting, recordIO: 0 })

            // set wav (input)
            const wavInput = document.getElementById(AUDIO_ELEMENT_FOR_SAMPLING_INPUT) as HTMLAudioElement
            wavInput.src = "/tmp/in.wav?" + new Date().getTime()
            wavInput.controls = true
            // @ts-ignore
            wavInput.setSinkId(audioOutputForAnalyzer)

            // set wav (output)
            const wavOutput = document.getElementById(AUDIO_ELEMENT_FOR_SAMPLING_OUTPUT) as HTMLAudioElement
            wavOutput.src = "/tmp/out.wav?" + new Date().getTime()
            wavOutput.controls = true
            // @ts-ignore
            wavOutput.setSinkId(audioOutputForAnalyzer)
        }

        const startClassName = serverIORecording ? "config-sub-area-buutton-active" : "config-sub-area-buutton"
        const stopClassName = serverIORecording ? "config-sub-area-buutton" : "config-sub-area-buutton-active"
        return (
            <>
                <div className="config-sub-area-control">
                    <div className="config-sub-area-control-title-long">ServerIO Analyzer</div>
                </div>

                <div className="config-sub-area-control left-padding-1">
                    <div className="config-sub-area-control-title">SIO rec.</div>
                    <div className="config-sub-area-control-field">
                        <div className="config-sub-area-buuttons">
                            <div onClick={onServerIORecordStartClicked} className={startClassName}>start</div>
                            <div onClick={onServerIORecordStopClicked} className={stopClassName}>stop</div>
                        </div>
                    </div>
                </div>


                <div className="config-sub-area-control left-padding-1">
                    <div className="config-sub-area-control-title">dev</div>
                    <div className="config-sub-area-control-field">
                        <div className="config-sub-area-control-field-auido-io">
                            <select className="body-select" value={audioOutputForAnalyzer} onChange={(e) => {
                                setAudioOutputForAnalyzer(e.target.value)
                                const wavInput = document.getElementById(AUDIO_ELEMENT_FOR_SAMPLING_INPUT) as HTMLAudioElement
                                const wavOutput = document.getElementById(AUDIO_ELEMENT_FOR_SAMPLING_OUTPUT) as HTMLAudioElement
                                //@ts-ignore
                                wavInput.setSinkId(e.target.value)
                                //@ts-ignore
                                wavOutput.setSinkId(e.target.value)
                            }}>
                                {
                                    outputAudioDeviceInfo.map(x => {
                                        if (x.deviceId == "none") {
                                            return null
                                        }
                                        return <option key={x.deviceId} value={x.deviceId}>{x.label}</option>
                                    }).filter(x => { return x != null })
                                }
                            </select>
                        </div>
                    </div>
                </div>

                <div className="config-sub-area-control left-padding-1">
                    <div className="config-sub-area-control-title">in</div>
                    <div className="config-sub-area-control-field">
                        <div className="config-sub-area-control-field-wav-file">
                            <div className="config-sub-area-control-field-wav-file-audio-container">
                                <audio className="config-sub-area-control-field-wav-file-audio" id={AUDIO_ELEMENT_FOR_SAMPLING_INPUT} controls></audio>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="config-sub-area-control left-padding-1">
                    <div className="config-sub-area-control-title">out</div>
                    <div className="config-sub-area-control-field">
                        <div className="config-sub-area-control-field-wav-file">
                            <div className="config-sub-area-control-field-wav-file-audio-container">
                                <audio className="config-sub-area-control-field-wav-file-audio" id={AUDIO_ELEMENT_FOR_SAMPLING_OUTPUT} controls></audio>
                            </div>
                        </div>
                    </div>
                </div>

            </>
        )

    }, [serverIORecording, workletNodeSetting])

    return (
        <div className="config-sub-area">
            {serverIORecorderRow}
        </div>
    )
}


