import { ServerInfoSoVitsSVC } from "@dannadori/voice-changer-client-js";
import React, { useMemo, useState } from "react"
import { useAppState } from "./001_provider/001_AppStateProvider";
import { AnimationTypes, HeaderButton, HeaderButtonProps } from "./components/101_HeaderButton";

export const useSpeakerSetting = () => {
    const appState = useAppState()


    const accodionButton = useMemo(() => {
        const accodionButtonProps: HeaderButtonProps = {
            stateControlCheckbox: appState.frontendManagerState.stateControls.openSpeakerSettingCheckbox,
            tooltip: "Open/Close",
            onIcon: ["fas", "caret-up"],
            offIcon: ["fas", "caret-up"],
            animation: AnimationTypes.spinner,
            tooltipClass: "tooltip-right",
        };
        return <HeaderButton {...accodionButtonProps}></HeaderButton>;
    }, []);

    const dstIdRow = useMemo(() => {
        const settings = appState.serverSetting.serverSetting as ServerInfoSoVitsSVC
        const speakers = settings.speakers
        if (!speakers) {
            return <></>
        }

        const currentValue = Object.values(speakers).includes(appState.serverSetting.serverSetting.dstId) ? appState.serverSetting.serverSetting.dstId : -1

        return (
            <div className="body-row split-3-2-1-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Destination Speaker Id</div>
                <div className="body-select-container">
                    <select className="body-select" value={currentValue} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, dstId: Number(e.target.value) })

                    }}>
                        <option key="unknown" value={0}>default(0)</option>
                        {
                            Object.keys(speakers).map(x => {
                                return <option key={x} value={speakers[x]}>{x}({speakers[x]})</option>
                            })
                        }
                    </select>
                </div>
                <div className="body-item-text">
                </div>
                <div className="body-item-text"></div>
            </div>
        )
    }, [appState.serverSetting.serverSetting, appState.serverSetting.updateServerSettings])


    const tranRow = useMemo(() => {
        return (
            <div className="body-row split-3-2-2-3 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Tran</div>
                <div>
                    <input type="range" className="body-item-input-slider" min="-50" max="50" step="1" value={appState.serverSetting.serverSetting.tran} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, tran: Number(e.target.value) })
                    }}></input>
                    <span className="body-item-input-slider-val">{appState.serverSetting.serverSetting.tran}</span>
                </div>
                <div>
                    <input type="checkbox" checked={appState.serverSetting.serverSetting.predictF0 == 1} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, predictF0: e.target.checked ? 1 : 0 })
                    }} /> predict f0
                </div>
                <div className="body-button-container">
                </div>
            </div>
        )
    }, [
        appState.serverSetting.serverSetting,
        appState.serverSetting.updateServerSettings
    ])


    const clusterInferRatioRow = useMemo(() => {
        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Cluster infer ratio</div>
                <div>
                    <input type="range" className="body-item-input-slider" min="0" max="1" step="0.1" value={appState.serverSetting.serverSetting.clusterInferRatio} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, clusterInferRatio: Number(e.target.value) })
                    }}></input>
                    <span className="body-item-input-slider-val">{appState.serverSetting.serverSetting.clusterInferRatio}</span>
                </div>
                <div className="body-button-container">
                </div>
            </div>
        )
    }, [
        appState.serverSetting.serverSetting,
        appState.serverSetting.updateServerSettings
    ])


    const noiceScaleRow = useMemo(() => {
        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Noice Scale</div>
                <div>
                    <input type="range" className="body-item-input-slider" min="0" max="1" step="0.1" value={appState.serverSetting.serverSetting.noiceScale} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, noiceScale: Number(e.target.value) })
                    }}></input>
                    <span className="body-item-input-slider-val">{appState.serverSetting.serverSetting.noiceScale}</span>
                </div>
                <div className="body-button-container">
                </div>
            </div>
        )
    }, [
        appState.serverSetting.serverSetting,
        appState.serverSetting.updateServerSettings
    ])

    const silentThresholdRow = useMemo(() => {
        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Silent Threshold</div>
                <div>
                    <input type="range" className="body-item-input-slider" min="0.00000" max="0.001" step="0.00001" value={appState.serverSetting.serverSetting.silentThreshold} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, silentThreshold: Number(e.target.value) })
                    }}></input>
                    <span className="body-item-input-slider-val">{appState.serverSetting.serverSetting.silentThreshold}</span>
                </div>

                <div className="body-button-container">
                </div>
            </div>
        )
    }, [
        appState.serverSetting.serverSetting,
        appState.serverSetting.updateServerSettings
    ])

    const speakerSetting = useMemo(() => {
        return (
            <>
                {appState.frontendManagerState.stateControls.openSpeakerSettingCheckbox.trigger}
                <div className="partition">
                    <div className="partition-header">
                        <span className="caret">
                            {accodionButton}
                        </span>
                        <span className="title" onClick={() => { appState.frontendManagerState.stateControls.openSpeakerSettingCheckbox.updateState(!appState.frontendManagerState.stateControls.openSpeakerSettingCheckbox.checked()) }}>
                            Speaker Setting
                        </span>
                    </div>

                    <div className="partition-content">
                        {/* {dstIdRow} */}
                        {tranRow}
                        {clusterInferRatioRow}
                        {/* {noiceScaleRow} */}
                        {silentThresholdRow}
                    </div>
                </div>
            </>
        )
    }, [dstIdRow, tranRow, noiceScaleRow, silentThresholdRow])

    return {
        speakerSetting,
    }

}


