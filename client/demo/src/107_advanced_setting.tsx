import { BufferSize, CrossFadeOverlapSize, DownSamplingMode, InputSampleRate, Protocol, SampleRate } from "@dannadori/voice-changer-client-js"
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
                    <input type="text" defaultValue={appState.streamerSetting.audioStreamerSetting.serverUrl} id="mmvc-server-url" className="body-item-input" />
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onSetServerClicked}>set</div>
                </div>
            </div>
        )
    }, [appState.streamerSetting.audioStreamerSetting.serverUrl, appState.clientSetting.setServerUrl])

    const protocolRow = useMemo(() => {
        const onProtocolChanged = async (val: Protocol) => {
            appState.streamerSetting.updateAudioStreamerSetting({ ...appState.streamerSetting.audioStreamerSetting, protocol: val })
        }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Protocol</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.streamerSetting.audioStreamerSetting.protocol} onChange={(e) => {
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
    }, [appState.streamerSetting.audioStreamerSetting.protocol, appState.streamerSetting.updateAudioStreamerSetting])


    const sampleRateRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Sample Rate</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.clientSetting.clientSetting.sampleRate} onChange={(e) => {
                        appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, sampleRate: Number(e.target.value) as SampleRate })
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
    }, [appState.clientSetting.clientSetting.sampleRate, appState.clientSetting.updateClientSetting])

    const sendingSampleRateRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Sending Sample Rate</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.streamerSetting.audioStreamerSetting.sendingSampleRate} onChange={(e) => {
                        appState.streamerSetting.updateAudioStreamerSetting({ ...appState.streamerSetting.audioStreamerSetting, sendingSampleRate: Number(e.target.value) as InputSampleRate })
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, inputSampleRate: Number(e.target.value) as InputSampleRate })
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
    }, [appState.streamerSetting.audioStreamerSetting.sendingSampleRate, appState.streamerSetting.updateAudioStreamerSetting, appState.serverSetting.updateServerSettings])

    const bufferSizeRow = useMemo(() => {
        return (

            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Buffer Size</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.clientSetting.clientSetting.bufferSize} onChange={(e) => {
                        appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, bufferSize: Number(e.target.value) as BufferSize })
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
    }, [appState.clientSetting.clientSetting.bufferSize, appState.clientSetting.updateClientSetting])


    const crossFadeOverlapSizeRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title  left-padding-1">Cross Fade Overlap Size</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.serverSetting.serverSetting.crossFadeOverlapSize} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, crossFadeOverlapSize: Number(e.target.value) as CrossFadeOverlapSize })
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
    }, [appState.serverSetting.serverSetting.crossFadeOverlapSize, appState.serverSetting.updateServerSettings])

    const crossFadeOffsetRateRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title  left-padding-1">Cross Fade Offset Rate</div>
                <div className="body-input-container">
                    <input type="number" min={0} max={1} step={0.1} value={appState.serverSetting.serverSetting.crossFadeOffsetRate} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, crossFadeOffsetRate: Number(e.target.value) })
                    }} />
                </div>
            </div>
        )
    }, [appState.serverSetting.serverSetting.crossFadeOffsetRate, appState.serverSetting.updateServerSettings])

    const crossFadeEndRateRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Cross Fade End Rate</div>
                <div className="body-input-container">
                    <input type="number" min={0} max={1} step={0.1} value={appState.serverSetting.serverSetting.crossFadeEndRate} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, crossFadeEndRate: Number(e.target.value) })
                    }} />
                </div>
            </div>
        )
    }, [appState.serverSetting.serverSetting.crossFadeEndRate, appState.serverSetting.updateServerSettings])


    const downSamplingModeRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">DownSamplingMode</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.streamerSetting.audioStreamerSetting.downSamplingMode} onChange={(e) => {
                        appState.streamerSetting.updateAudioStreamerSetting({ ...appState.streamerSetting.audioStreamerSetting, downSamplingMode: e.target.value as DownSamplingMode })
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
    }, [appState.streamerSetting.audioStreamerSetting.downSamplingMode, appState.streamerSetting.updateAudioStreamerSetting])


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
                {crossFadeOverlapSizeRow}
                {crossFadeOffsetRateRow}
                {crossFadeEndRateRow}
                <div className="body-row divider"></div>
                {workletSettingRow}
                <div className="body-row divider"></div>
                {downSamplingModeRow}

            </>
        )
    }, [mmvcServerUrlRow, protocolRow, sampleRateRow, sendingSampleRateRow, bufferSizeRow, crossFadeOverlapSizeRow, crossFadeOffsetRateRow, crossFadeEndRateRow, workletSettingRow, downSamplingModeRow])


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


