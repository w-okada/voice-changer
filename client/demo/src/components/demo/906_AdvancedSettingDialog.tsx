import React, { useMemo } from "react";
import { useGuiState } from "./001_GuiStateProvider";
import { useAppState } from "../../001_provider/001_AppStateProvider";
import { Protocol } from "@dannadori/voice-changer-client-js";

export const AdvancedSettingDialog = () => {
    const guiState = useGuiState();
    const { setting, serverSetting, setWorkletNodeSetting, setWorkletSetting, setVoiceChangerClientSetting } = useAppState();
    const dialog = useMemo(() => {
        const closeButtonRow = (
            <div className="body-row split-3-4-3 left-padding-1">
                <div className="body-item-text"></div>
                <div className="body-button-container body-button-container-space-around">
                    <div
                        className="body-button"
                        onClick={() => {
                            guiState.stateControls.showAdvancedSettingCheckbox.updateState(false);
                        }}
                    >
                        close
                    </div>
                </div>
                <div className="body-item-text"></div>
            </div>
        );

        const onProtocolChanged = async (val: Protocol) => {
            setWorkletNodeSetting({ ...setting.workletNodeSetting, protocol: val });
        };
        const protocolRow = (
            <div className="advanced-setting-container-row">
                <div className="advanced-setting-container-row-title">protocol</div>
                <div className="advanced-setting-container-row-field">
                    <select
                        value={setting.workletNodeSetting.protocol}
                        onChange={(e) => {
                            onProtocolChanged(e.target.value as Protocol);
                        }}
                    >
                        {Object.values(Protocol).map((x) => {
                            return (
                                <option key={x} value={x}>
                                    {x}
                                </option>
                            );
                        })}
                    </select>
                </div>
            </div>
        );
        const crossfaceRow = (
            <div className="advanced-setting-container-row">
                <div className="advanced-setting-container-row-title">Crossfade length</div>
                <div className="advanced-setting-container-row-field">
                    <div>
                        <input
                            type="range"
                            className="body-item-input-slider"
                            min="0.05"
                            max="0.20"
                            step="0.01"
                            value={serverSetting.serverSetting.crossFadeOverlapSize}
                            onChange={(e) => {
                                serverSetting.updateServerSettings({ ...serverSetting.serverSetting, crossFadeOverlapSize: Number(e.target.value) });
                            }}
                        ></input>
                        <span className="body-item-input-slider-val">{serverSetting.serverSetting.crossFadeOverlapSize} ms</span>
                    </div>
                </div>
            </div>
        );

        const onSilenceFrontChanged = (val: number) => {
            serverSetting.updateServerSettings({
                ...serverSetting.serverSetting,
                silenceFront: val,
            });
        };
        const silenceFrontRow = (
            <div className="advanced-setting-container-row">
                <div className="advanced-setting-container-row-title">SilenceFront</div>
                <div className="advanced-setting-container-row-field">
                    <select
                        value={serverSetting.serverSetting.silenceFront}
                        onChange={(e) => {
                            onSilenceFrontChanged(Number(e.target.value));
                        }}
                    >
                        <option value="0">off</option>
                        <option value="1">on</option>
                    </select>
                </div>
            </div>
        );

        const onForceFp32ModeChanged = (val: number) => {
            serverSetting.updateServerSettings({
                ...serverSetting.serverSetting,
                forceFp32: val,
            });
        };
        const forceFp32ModeRow = (
            <div className="advanced-setting-container-row">
                <div className="advanced-setting-container-row-title">Force FP32 mode</div>
                <div className="advanced-setting-container-row-field">
                    <select
                        value={serverSetting.serverSetting.forceFp32}
                        onChange={(e) => {
                            onForceFp32ModeChanged(Number(e.target.value));
                        }}
                    >
                        <option value="0">off</option>
                        <option value="1">on</option>
                    </select>
                </div>
            </div>
        );

        const protectRow = (
            <div className="advanced-setting-container-row">
                <div className="advanced-setting-container-row-title">Protect</div>
                <div className="advanced-setting-container-row-field">
                    <div>
                        <input
                            type="range"
                            className="body-item-input-slider"
                            min="0"
                            max="0.5"
                            step="0.01"
                            value={serverSetting.serverSetting.protect || 0}
                            onChange={(e) => {
                                serverSetting.updateServerSettings({ ...serverSetting.serverSetting, protect: Number(e.target.value) });
                            }}
                        ></input>
                        <span className="body-item-input-slider-val">{serverSetting.serverSetting.protect}</span>
                    </div>
                </div>
            </div>
        );

        const onRVCQualityChanged = (val: number) => {
            serverSetting.updateServerSettings({
                ...serverSetting.serverSetting,
                rvcQuality: val,
            });
        };
        const rvcQualityRow = (
            <div className="advanced-setting-container-row">
                <div className="advanced-setting-container-row-title">RVC Quality</div>
                <div className="advanced-setting-container-row-field">
                    <select
                        value={serverSetting.serverSetting.rvcQuality}
                        onChange={(e) => {
                            onRVCQualityChanged(Number(e.target.value));
                        }}
                    >
                        <option value="0">low</option>
                        <option value="1">high</option>
                    </select>
                </div>
            </div>
        );
        const skipPassThroughConfirmationRow = (
            <div className="advanced-setting-container-row">
                <div className="advanced-setting-container-row-title-long">Skip Pass through confirmation</div>
                <div className="advanced-setting-container-row-field">
                    <select
                        value={setting.voiceChangerClientSetting.passThroughConfirmationSkip ? "1" : "0"}
                        onChange={(e) => {
                            setVoiceChangerClientSetting({ ...setting.voiceChangerClientSetting, passThroughConfirmationSkip: e.target.value == "1" ? true : false });
                        }}
                    >
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
            </div>
        );
        const content = (
            <div className="advanced-setting-container">
                {protocolRow}
                {crossfaceRow}
                {silenceFrontRow}
                {forceFp32ModeRow}
                {protectRow}
                {rvcQualityRow}
                {skipPassThroughConfirmationRow}
            </div>
        );

        return (
            <div className="dialog-frame">
                <div className="dialog-title">Advanced Setting</div>
                <div className="dialog-content">
                    {content}
                    {closeButtonRow}
                </div>
            </div>
        );
    }, [serverSetting.serverSetting, serverSetting.updateServerSettings, setting.workletNodeSetting, setWorkletNodeSetting, setting.workletSetting, setWorkletSetting]);
    return dialog;
};
