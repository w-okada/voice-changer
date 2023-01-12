import { BufferSize, Protocol, SampleRate, VoiceChangerMode } from "@dannadori/voice-changer-client-js"
import React, { useMemo, useState } from "react"
import { ClientState } from "@dannadori/voice-changer-client-js";


export type UseAdvancedSettingProps = {
    clientState: ClientState
}

export type AdvancedSettingState = {
    advancedSetting: JSX.Element;
}

export const useAdvancedSetting = (props: UseAdvancedSettingProps): AdvancedSettingState => {
    const [showAdvancedSetting, setShowAdvancedSetting] = useState<boolean>(false)
    const mmvcServerUrlRow = useMemo(() => {
        const onSetServerClicked = async () => {
            const input = document.getElementById("mmvc-server-url") as HTMLInputElement
            props.clientState.clientSetting.setServerUrl(input.value)
        }
        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">MMVC Server</div>
                <div className="body-input-container">
                    <input type="text" defaultValue={props.clientState.clientSetting.setting.mmvcServerUrl} id="mmvc-server-url" className="body-item-input" />
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onSetServerClicked}>set</div>
                </div>
            </div>
        )
    }, [props.clientState.clientSetting.setting.mmvcServerUrl, props.clientState.clientSetting.setServerUrl])

    const protocolRow = useMemo(() => {
        const onProtocolChanged = async (val: Protocol) => {
            props.clientState.clientSetting.setProtocol(val)
        }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Protocol</div>
                <div className="body-select-container">
                    <select className="body-select" value={props.clientState.clientSetting.setting.protocol} onChange={(e) => {
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
        )
    }, [props.clientState.clientSetting.setting.protocol, props.clientState.clientSetting.setProtocol])


    const sampleRateRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Sample Rate</div>
                <div className="body-select-container">
                    <select className="body-select" value={props.clientState.clientSetting.setting.sampleRate} onChange={(e) => {
                        props.clientState.clientSetting.setSampleRate(Number(e.target.value) as SampleRate)
                    }}>
                        {
                            Object.values(SampleRate).map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [props.clientState.clientSetting.setting.sampleRate, props.clientState.clientSetting.setSampleRate])

    const bufferSizeRow = useMemo(() => {
        return (

            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Buffer Size</div>
                <div className="body-select-container">
                    <select className="body-select" value={props.clientState.clientSetting.setting.bufferSize} onChange={(e) => {
                        props.clientState.clientSetting.setBufferSize(Number(e.target.value) as BufferSize)
                    }}>
                        {
                            Object.values(BufferSize).map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [props.clientState.clientSetting.setting.bufferSize, props.clientState.clientSetting.setBufferSize])

    const convertChunkNumRow = useMemo(() => {
        return (

            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Convert Chunk Num(128sample/chunk)</div>
                <div className="body-input-container">
                    <input type="number" min={1} max={256} step={1} value={props.clientState.serverSetting.setting.convertChunkNum} onChange={(e) => {
                        props.clientState.serverSetting.setConvertChunkNum(Number(e.target.value))
                    }} />
                </div>
            </div>
        )
    }, [props.clientState.serverSetting.setting.convertChunkNum, props.clientState.serverSetting.setConvertChunkNum])

    const minConvertSizeRow = useMemo(() => {
        return (

            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Min Convert Size(byte)</div>
                <div className="body-input-container">
                    <input type="number" min={0} max={8196} step={8196} value={props.clientState.serverSetting.setting.minConvertSize} onChange={(e) => {
                        props.clientState.serverSetting.setMinConvertSize(Number(e.target.value))
                    }} />
                </div>
            </div>
        )
    }, [props.clientState.serverSetting.setting.minConvertSize, props.clientState.serverSetting.setMinConvertSize])

    const crossFadeOverlapRateRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title  left-padding-1">Cross Fade Overlap Rate</div>
                <div className="body-input-container">
                    <input type="number" min={0.1} max={1} step={0.1} value={props.clientState.serverSetting.setting.crossFadeOverlapRate} onChange={(e) => {
                        props.clientState.serverSetting.setCrossFadeOverlapRate(Number(e.target.value))
                    }} />
                </div>
            </div>
        )
    }, [props.clientState.serverSetting.setting.crossFadeOverlapRate, props.clientState.serverSetting.setCrossFadeOverlapRate])

    const crossFadeOffsetRateRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title  left-padding-1">Cross Fade Offset Rate</div>
                <div className="body-input-container">
                    <input type="number" min={0} max={1} step={0.1} value={props.clientState.serverSetting.setting.crossFadeOffsetRate} onChange={(e) => {
                        props.clientState.serverSetting.setCrossFadeOffsetRate(Number(e.target.value))
                    }} />
                </div>
            </div>
        )
    }, [props.clientState.serverSetting.setting.crossFadeOffsetRate, props.clientState.serverSetting.setCrossFadeOffsetRate])

    const crossFadeEndRateRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Cross Fade End Rate</div>
                <div className="body-input-container">
                    <input type="number" min={0} max={1} step={0.1} value={props.clientState.serverSetting.setting.crossFadeEndRate} onChange={(e) => {
                        props.clientState.serverSetting.setCrossFadeEndRate(Number(e.target.value))
                    }} />
                </div>
            </div>
        )
    }, [props.clientState.serverSetting.setting.crossFadeEndRate, props.clientState.serverSetting.setCrossFadeEndRate])


    const vfForceDisableRow = useMemo(() => {
        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">VF Disabled</div>
                <div>
                    <input type="checkbox" checked={props.clientState.clientSetting.setting.forceVfDisable} onChange={(e) => {
                        props.clientState.clientSetting.setVfForceDisabled(e.target.checked)
                    }} />
                </div>
                <div className="body-button-container">
                </div>
            </div>
        )
    }, [props.clientState.clientSetting.setting.forceVfDisable, props.clientState.clientSetting.setVfForceDisabled])

    const voiceChangeModeRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Voice Change Mode</div>
                <div className="body-select-container">
                    <select className="body-select" value={props.clientState.clientSetting.setting.voiceChangerMode} onChange={(e) => {
                        props.clientState.clientSetting.setVoiceChangerMode(e.target.value as VoiceChangerMode)
                    }}>
                        {
                            Object.values(VoiceChangerMode).map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [props.clientState.clientSetting.setting.voiceChangerMode, props.clientState.clientSetting.setVoiceChangerMode])



    const workletSettingRow = useMemo(() => {
        return (
            <>

                <div className="body-row split-3-7 left-padding-1 guided">
                    <div className="body-item-title left-padding-1">Trancate Num</div>
                    <div className="body-input-container">
                        <input type="number" min={50} max={300} step={1} value={props.clientState.workletSetting.setting.numTrancateTreshold} onChange={(e) => {
                            props.clientState.workletSetting.setSetting({
                                ...props.clientState.workletSetting.setting,
                                numTrancateTreshold: Number(e.target.value)
                            })
                        }} />
                    </div>
                </div>

                <div className="body-row split-3-7 left-padding-1 guided">
                    <div className="body-item-title left-padding-1">Trancate Vol</div>
                    <div className="body-input-container">
                        <input type="number" min={0.0001} max={0.0009} step={0.0001} value={props.clientState.workletSetting.setting.volTrancateThreshold} onChange={(e) => {
                            props.clientState.workletSetting.setSetting({
                                ...props.clientState.workletSetting.setting,
                                volTrancateThreshold: Number(e.target.value)
                            })
                        }} />
                    </div>
                </div>
                <div className="body-row split-3-7 left-padding-1 guided">
                    <div className="body-item-title left-padding-1">Trancate Vol Length</div>
                    <div className="body-input-container">
                        <input type="number" min={16} max={128} step={1} value={props.clientState.workletSetting.setting.volTrancateLength} onChange={(e) => {
                            props.clientState.workletSetting.setSetting({
                                ...props.clientState.workletSetting.setting,
                                volTrancateLength: Number(e.target.value)
                            })
                        }} />
                    </div>
                </div>
            </>
        )
    }, [props.clientState.workletSetting.setting, props.clientState.workletSetting.setSetting])


    const advanceSettingContent = useMemo(() => {
        if (!showAdvancedSetting) return <></>
        return (
            <>
                <div className="body-row divider"></div>
                {mmvcServerUrlRow}
                {protocolRow}
                <div className="body-row divider"></div>
                {sampleRateRow}
                {bufferSizeRow}
                <div className="body-row divider"></div>

                {convertChunkNumRow}
                {minConvertSizeRow}
                {crossFadeOverlapRateRow}
                {crossFadeOffsetRateRow}
                {crossFadeEndRateRow}
                <div className="body-row divider"></div>
                {vfForceDisableRow}
                {voiceChangeModeRow}
                <div className="body-row divider"></div>
                {workletSettingRow}
                <div className="body-row divider"></div>
            </>
        )
    }, [showAdvancedSetting, mmvcServerUrlRow, protocolRow, sampleRateRow, bufferSizeRow, convertChunkNumRow, minConvertSizeRow, crossFadeOverlapRateRow, crossFadeOffsetRateRow, crossFadeEndRateRow, vfForceDisableRow, voiceChangeModeRow, workletSettingRow])


    const advancedSetting = useMemo(() => {
        return (
            <>
                <div className="body-row split-3-7 left-padding-1">
                    <div className="body-sub-section-title">Advanced Setting</div>
                    <div>
                        <input type="checkbox" checked={showAdvancedSetting} onChange={(e) => {
                            setShowAdvancedSetting(e.target.checked)
                        }} /> show
                    </div>
                </div>
                {advanceSettingContent}
            </>
        )
    }, [showAdvancedSetting, advanceSettingContent])

    return {
        advancedSetting,
    }

}


