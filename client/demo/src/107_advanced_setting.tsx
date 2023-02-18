import { BufferSize, CrossFadeOverlapSize, DownSamplingMode, InputSampleRate, Protocol, SampleRate, VoiceChangerMode } from "@dannadori/voice-changer-client-js"
import React, { useMemo } from "react"
import { useAppState } from "./001_provider/001_AppStateProvider";
import { AnimationTypes, HeaderButton, HeaderButtonProps } from "./components/101_HeaderButton";

export type AdvancedSettingState = {
    advancedSetting: JSX.Element;
}

export const useAdvancedSetting = (): AdvancedSettingState => {
    const appState = useAppState()
    const accodionButton = useMemo(() => {
        const accodionButtonProps: HeaderButtonProps = {
            stateControlCheckbox: appState.frontendManagerState.stateControls.openAdvancedSettingCheckbox,
            tooltip: "Open/Close",
            onIcon: ["fas", "caret-up"],
            offIcon: ["fas", "caret-up"],
            animation: AnimationTypes.spinner,
            tooltipClass: "tooltip-right",
        };
        return <HeaderButton {...accodionButtonProps}></HeaderButton>;
    }, []);

    const mmvcServerUrlRow = useMemo(() => {
        const onSetServerClicked = async () => {
            const input = document.getElementById("mmvc-server-url") as HTMLInputElement
            appState.clientSetting.setServerUrl(input.value)
        }
        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">MMVC Server</div>
                <div className="body-input-container">
                    <input type="text" defaultValue={appState.clientSetting.setting.mmvcServerUrl} id="mmvc-server-url" className="body-item-input" />
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onSetServerClicked}>set</div>
                </div>
            </div>
        )
    }, [appState.clientSetting.setting.mmvcServerUrl, appState.clientSetting.setServerUrl])

    const protocolRow = useMemo(() => {
        const onProtocolChanged = async (val: Protocol) => {
            appState.clientSetting.setProtocol(val)
        }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Protocol</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.clientSetting.setting.protocol} onChange={(e) => {
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
    }, [appState.clientSetting.setting.protocol, appState.clientSetting.setProtocol])


    const sampleRateRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Sample Rate</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.clientSetting.setting.sampleRate} onChange={(e) => {
                        appState.clientSetting.setSampleRate(Number(e.target.value) as SampleRate)
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
    }, [appState.clientSetting.setting.sampleRate, appState.clientSetting.setSampleRate])

    const sendingSampleRateRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Sending Sample Rate</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.clientSetting.setting.sendingSampleRate} onChange={(e) => {
                        appState.clientSetting.setSendingSampleRate(Number(e.target.value) as InputSampleRate)
                        appState.serverSetting.setInputSampleRate(Number(e.target.value) as InputSampleRate)

                    }}>
                        {
                            Object.values(InputSampleRate).map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [appState.clientSetting.setting.sendingSampleRate, appState.clientSetting.setSendingSampleRate, appState.serverSetting.setInputSampleRate])

    const bufferSizeRow = useMemo(() => {
        return (

            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Buffer Size</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.clientSetting.setting.bufferSize} onChange={(e) => {
                        appState.clientSetting.setBufferSize(Number(e.target.value) as BufferSize)
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
    }, [appState.clientSetting.setting.bufferSize, appState.clientSetting.setBufferSize])

    const convertChunkNumRow = useMemo(() => {
        return (

            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Convert Chunk Num(128sample/chunk)</div>
                <div className="body-input-container">
                    <input type="number" min={1} max={256} step={1} value={appState.serverSetting.setting.convertChunkNum} onChange={(e) => {
                        appState.serverSetting.setConvertChunkNum(Number(e.target.value))
                    }} />
                </div>
            </div>
        )
    }, [appState.serverSetting.setting.convertChunkNum, appState.serverSetting.setConvertChunkNum])

    const minConvertSizeRow = useMemo(() => {
        return (

            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Min Convert Size(byte)</div>
                <div className="body-input-container">
                    <input type="number" min={0} max={8196} step={8196} value={appState.serverSetting.setting.minConvertSize} onChange={(e) => {
                        appState.serverSetting.setMinConvertSize(Number(e.target.value))
                    }} />
                </div>
            </div>
        )
    }, [appState.serverSetting.setting.minConvertSize, appState.serverSetting.setMinConvertSize])

    const crossFadeOverlapRateRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title  left-padding-1">Cross Fade Overlap Rate</div>
                <div className="body-input-container">
                    <input type="number" min={0.1} max={1} step={0.1} value={appState.serverSetting.setting.crossFadeOverlapRate} onChange={(e) => {
                        appState.serverSetting.setCrossFadeOverlapRate(Number(e.target.value))
                    }} />
                </div>
            </div>
        )
    }, [appState.serverSetting.setting.crossFadeOverlapRate, appState.serverSetting.setCrossFadeOverlapRate])


    const crossFadeOverlapSizeRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title  left-padding-1">Cross Fade Overlap Size</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.serverSetting.setting.crossFadeOverlapSize} onChange={(e) => {
                        appState.serverSetting.setCrossFadeOverlapSize(Number(e.target.value) as CrossFadeOverlapSize)
                    }}>
                        {
                            Object.values(CrossFadeOverlapSize).map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [appState.serverSetting.setting.crossFadeOverlapSize, appState.serverSetting.setCrossFadeOverlapSize])

    const crossFadeOffsetRateRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title  left-padding-1">Cross Fade Offset Rate</div>
                <div className="body-input-container">
                    <input type="number" min={0} max={1} step={0.1} value={appState.serverSetting.setting.crossFadeOffsetRate} onChange={(e) => {
                        appState.serverSetting.setCrossFadeOffsetRate(Number(e.target.value))
                    }} />
                </div>
            </div>
        )
    }, [appState.serverSetting.setting.crossFadeOffsetRate, appState.serverSetting.setCrossFadeOffsetRate])

    const crossFadeEndRateRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Cross Fade End Rate</div>
                <div className="body-input-container">
                    <input type="number" min={0} max={1} step={0.1} value={appState.serverSetting.setting.crossFadeEndRate} onChange={(e) => {
                        appState.serverSetting.setCrossFadeEndRate(Number(e.target.value))
                    }} />
                </div>
            </div>
        )
    }, [appState.serverSetting.setting.crossFadeEndRate, appState.serverSetting.setCrossFadeEndRate])


    const voiceChangeModeRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Voice Change Mode</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.clientSetting.setting.voiceChangerMode} onChange={(e) => {
                        appState.clientSetting.setVoiceChangerMode(e.target.value as VoiceChangerMode)
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
    }, [appState.clientSetting.setting.voiceChangerMode, appState.clientSetting.setVoiceChangerMode])


    const downSamplingModeRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">DownSamplingMode</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.clientSetting.setting.downSamplingMode} onChange={(e) => {
                        appState.clientSetting.setDownSamplingMode(e.target.value as DownSamplingMode)
                    }}>
                        {
                            Object.values(DownSamplingMode).map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [appState.clientSetting.setting.downSamplingMode, appState.clientSetting.setDownSamplingMode])


    const workletSettingRow = useMemo(() => {
        return (
            <>

                <div className="body-row split-3-7 left-padding-1 guided">
                    <div className="body-item-title left-padding-1">Trancate Num</div>
                    <div className="body-input-container">
                        <input type="number" min={5} max={300} step={1} value={appState.workletSetting.setting.numTrancateTreshold} onChange={(e) => {
                            appState.workletSetting.setSetting({
                                ...appState.workletSetting.setting,
                                numTrancateTreshold: Number(e.target.value)
                            })
                        }} />
                    </div>
                </div>

                {/* v.1.5.xより Silent skipは廃止 */}
                {/* <div className="body-row split-3-7 left-padding-1 guided">
                    <div className="body-item-title left-padding-1">Trancate Vol</div>
                    <div className="body-input-container">
                        <input type="number" min={0.0001} max={0.0009} step={0.0001} value={appState.workletSetting.setting.volTrancateThreshold} onChange={(e) => {
                            appState.workletSetting.setSetting({
                                ...appState.workletSetting.setting,
                                volTrancateThreshold: Number(e.target.value)
                            })
                        }} />
                    </div>
                </div>
                <div className="body-row split-3-7 left-padding-1 guided">
                    <div className="body-item-title left-padding-1">Trancate Vol Length</div>
                    <div className="body-input-container">
                        <input type="number" min={16} max={128} step={1} value={appState.workletSetting.setting.volTrancateLength} onChange={(e) => {
                            appState.workletSetting.setSetting({
                                ...appState.workletSetting.setting,
                                volTrancateLength: Number(e.target.value)
                            })
                        }} />
                    </div>
                </div> */}
            </>
        )
    }, [appState.workletSetting.setting, appState.workletSetting.setSetting])


    const advanceSettingContent = useMemo(() => {
        return (
            <>
                <div className="body-row divider"></div>
                {mmvcServerUrlRow}
                {protocolRow}
                <div className="body-row divider"></div>
                {sampleRateRow}
                {sendingSampleRateRow}
                {bufferSizeRow}
                <div className="body-row divider"></div>

                {convertChunkNumRow}
                {minConvertSizeRow}
                {crossFadeOverlapRateRow}
                {crossFadeOverlapSizeRow}
                {crossFadeOffsetRateRow}
                {crossFadeEndRateRow}
                <div className="body-row divider"></div>
                {voiceChangeModeRow}
                <div className="body-row divider"></div>
                {workletSettingRow}
                <div className="body-row divider"></div>
                {downSamplingModeRow}

            </>
        )
    }, [mmvcServerUrlRow, protocolRow, sampleRateRow, sendingSampleRateRow, bufferSizeRow, convertChunkNumRow, minConvertSizeRow, crossFadeOverlapRateRow, crossFadeOverlapSizeRow, crossFadeOffsetRateRow, crossFadeEndRateRow, voiceChangeModeRow, workletSettingRow, downSamplingModeRow])


    const advancedSetting = useMemo(() => {
        return (
            <>
                {appState.frontendManagerState.stateControls.openAdvancedSettingCheckbox.trigger}
                <div className="partition">
                    <div className="partition-header">
                        <span className="caret">
                            {accodionButton}
                        </span>
                        <span className="title" onClick={() => { appState.frontendManagerState.stateControls.openAdvancedSettingCheckbox.updateState(!appState.frontendManagerState.stateControls.openAdvancedSettingCheckbox.checked()) }}>
                            Advanced Setting
                        </span>
                    </div>

                    <div className="partition-content">
                        {advanceSettingContent}
                    </div>
                </div>
            </>
        )
    }, [advanceSettingContent])

    return {
        advancedSetting,
    }

}


