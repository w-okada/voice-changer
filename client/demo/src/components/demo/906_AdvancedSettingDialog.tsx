import React, { useMemo } from "react";
import { useGuiState } from "./001_GuiStateProvider";
import { useAppState } from "../../001_provider/001_AppStateProvider";
import { CrossFadeOverlapSize, Protocol } from "@dannadori/voice-changer-client-js";


export const AdvancedSettingDialog = () => {
    const guiState = useGuiState()
    const { workletNodeSetting, workletSetting, serverSetting } = useAppState()
    const dialog = useMemo(() => {
        const closeButtonRow = (
            <div className="body-row split-3-4-3 left-padding-1">
                <div className="body-item-text">
                </div>
                <div className="body-button-container body-button-container-space-around">
                    <div className="body-button" onClick={() => { guiState.stateControls.showAdvancedSettingCheckbox.updateState(false) }} >close</div>
                </div>
                <div className="body-item-text"></div>
            </div>

        )

        const onProtocolChanged = async (val: Protocol) => {
            workletNodeSetting.updateWorkletNodeSetting({ ...workletNodeSetting.workletNodeSetting, protocol: val })
        }
        const protocolRow = (
            <div className="advanced-setting-container-row">
                <div className="advanced-setting-container-row-title">
                    protocol
                </div>
                <div className="advanced-setting-container-row-field">
                    <select value={workletNodeSetting.workletNodeSetting.protocol} onChange={(e) => {
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
        const crossfaceRow = (
            <div className="advanced-setting-container-row">
                <div className="advanced-setting-container-row-title">
                    Crossfade
                </div>
                <div className="advanced-setting-container-row-field">
                    <div className="advanced-setting-container-row-field-crossfade-container">
                        <div>
                            <div>overlap:</div>
                            <div>
                                <select className="body-select" value={serverSetting.serverSetting.crossFadeOverlapSize} onChange={(e) => {
                                    serverSetting.updateServerSettings({ ...serverSetting.serverSetting, crossFadeOverlapSize: Number(e.target.value) as CrossFadeOverlapSize })
                                }}>
                                    {
                                        Object.values(CrossFadeOverlapSize).map(x => {
                                            return <option key={x} value={x}>{x}</option>
                                        })
                                    }
                                </select>
                            </div>
                        </div>
                        <div>
                            <div>start:</div>
                            <div>
                                <input type="number" min={0} max={1} step={0.1} value={serverSetting.serverSetting.crossFadeOffsetRate} onChange={(e) => {
                                    serverSetting.updateServerSettings({ ...serverSetting.serverSetting, crossFadeOffsetRate: Number(e.target.value) })
                                }} />
                            </div>
                        </div>
                        <div>
                            <div>end:</div>
                            <div>
                                <input type="number" min={0} max={1} step={0.1} value={serverSetting.serverSetting.crossFadeEndRate} onChange={(e) => {
                                    serverSetting.updateServerSettings({ ...serverSetting.serverSetting, crossFadeEndRate: Number(e.target.value) })
                                }} />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        )

        const trancateRow = (
            <div className="advanced-setting-container-row">
                <div className="advanced-setting-container-row-title">
                    Trancate
                </div>
                <div className="advanced-setting-container-row-field">
                    <input type="number" min={5} max={300} step={1} value={workletSetting.setting.numTrancateTreshold} onChange={(e) => {
                        workletSetting.setSetting({
                            ...workletSetting.setting,
                            numTrancateTreshold: Number(e.target.value)
                        })
                    }} />
                </div>
            </div>
        )

        const onSilenceFrontChanged = (val: number) => {
            serverSetting.updateServerSettings({
                ...serverSetting.serverSetting,
                silenceFront: val
            })
        }
        const silenceFrontRow = (
            <div className="advanced-setting-container-row">
                <div className="advanced-setting-container-row-title">
                    SilenceFront
                </div>
                <div className="advanced-setting-container-row-field">
                    <select value={serverSetting.serverSetting.silenceFront} onChange={(e) => { onSilenceFrontChanged(Number(e.target.value)) }}>
                        <option value="0" >off</option>
                        <option value="1" >on</option>
                    </select>
                </div>
            </div>
        )

        const protectRow = (
            <div className="advanced-setting-container-row">
                <div className="advanced-setting-container-row-title">
                    Protect
                </div>
                <div className="advanced-setting-container-row-field">
                    <div>
                        <input type="range" className="body-item-input-slider" min="0" max="0.5" step="0.1" value={serverSetting.serverSetting.protect || 0} onChange={(e) => {
                            serverSetting.updateServerSettings({ ...serverSetting.serverSetting, protect: Number(e.target.value) })
                        }}></input>
                        <span className="body-item-input-slider-val">{serverSetting.serverSetting.protect}</span>
                    </div>
                </div>
            </div>
        )

        const onRVCQualityChanged = (val: number) => {
            serverSetting.updateServerSettings({
                ...serverSetting.serverSetting,
                rvcQuality: val
            })
        }
        const rvcQualityRow = (
            <div className="advanced-setting-container-row">
                <div className="advanced-setting-container-row-title">
                    RVC Quality
                </div>
                <div className="advanced-setting-container-row-field">
                    <select value={serverSetting.serverSetting.rvcQuality} onChange={(e) => { onRVCQualityChanged(Number(e.target.value)) }}>
                        <option value="0" >low</option>
                        <option value="1" >high</option>
                    </select>
                </div>
            </div>
        )

        const content = (
            <div className="advanced-setting-container">
                {protocolRow}
                {crossfaceRow}
                {trancateRow}
                {silenceFrontRow}
                {protectRow}
                {rvcQualityRow}
            </div>
        )

        return (
            <div className="dialog-frame">
                <div className="dialog-title">Advanced Setting</div>
                <div className="dialog-content">
                    {content}
                    {closeButtonRow}
                </div>
            </div>
        );
    }, [serverSetting.serverSetting, serverSetting.updateServerSettings, workletNodeSetting.workletNodeSetting, workletNodeSetting.updateWorkletNodeSetting, workletSetting.setting, workletSetting.setSetting]);
    return dialog;

};
