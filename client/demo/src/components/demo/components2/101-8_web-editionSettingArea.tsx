import React, { useMemo } from "react";
import { useAppState } from "../../../001_provider/001_AppStateProvider";
import { useAppRoot } from "../../../001_provider/001_AppRootProvider";
import { useGuiState } from "../001_GuiStateProvider";

export type WebEditionSettingAreaProps = {};

export const WebEditionSettingArea = (_props: WebEditionSettingAreaProps) => {
    const { serverSetting, webInfoState } = useAppState();
    const { appGuiSettingState } = useAppRoot();
    const webEdition = appGuiSettingState.edition.indexOf("web") >= 0;
    const guiState = useGuiState();

    const selected = useMemo(() => {
        if (webEdition) {
            return webInfoState.webModelslot;
        }
        return null;
    }, [webEdition]);

    const settingArea = useMemo(() => {
        if (!selected) {
            return <></>;
        }

        const readyForConfig = guiState.isConverting == false && webInfoState.webModelLoadingState == "ready";

        const versionV1ClassName = "character-area-control-button" + (webInfoState.voiceChangerConfig.config.voiceChangerType == "rvcv1" ? " character-area-control-button-active" : " character-area-control-button-stanby");
        const versionV2ClassName = "character-area-control-button" + (webInfoState.voiceChangerConfig.config.voiceChangerType == "rvcv2" ? " character-area-control-button-active" : " character-area-control-button-stanby");
        const verison = (
            <div className="character-area-control">
                <div className="character-area-control-title">Version</div>
                <div className="character-area-control-field">
                    <div className="character-area-slider-control">
                        <span className="character-area-slider-control-kind"></span>
                        <span className="character-area-control-buttons">
                            <span
                                className={!readyForConfig ? "character-area-control-button-disable" : versionV1ClassName}
                                onClick={() => {
                                    if (webInfoState.voiceChangerConfig.config.voiceChangerType == "rvcv1" || !readyForConfig) return;
                                    webInfoState.setVoiceChangerConfig("rvcv1", webInfoState.voiceChangerConfig.sampleRate, webInfoState.voiceChangerConfig.useF0, webInfoState.voiceChangerConfig.inputLength);
                                }}
                            >
                                v1
                            </span>
                            <span
                                className={!readyForConfig ? "character-area-control-button-disable" : versionV2ClassName}
                                onClick={() => {
                                    if (webInfoState.voiceChangerConfig.config.voiceChangerType == "rvcv2" || !readyForConfig) return;
                                    webInfoState.setVoiceChangerConfig("rvcv2", webInfoState.voiceChangerConfig.sampleRate, webInfoState.voiceChangerConfig.useF0, webInfoState.voiceChangerConfig.inputLength);
                                }}
                            >
                                v2
                            </span>
                        </span>
                    </div>
                </div>
            </div>
        );

        const sr32KClassName = "character-area-control-button" + (webInfoState.voiceChangerConfig.sampleRate == "32k" ? " character-area-control-button-active" : " character-area-control-button-stanby");
        const sr40KClassName = "character-area-control-button" + (webInfoState.voiceChangerConfig.sampleRate == "40k" ? " character-area-control-button-active" : " character-area-control-button-stanby");
        const sampleRate = (
            <div className="character-area-control">
                <div className="character-area-control-title">SR</div>
                <div className="character-area-control-field">
                    <div className="character-area-slider-control">
                        <span className="character-area-slider-control-kind"></span>
                        <span className="character-area-control-buttons">
                            <span
                                className={!readyForConfig ? "character-area-control-button-disable" : sr32KClassName}
                                onClick={() => {
                                    if (webInfoState.voiceChangerConfig.sampleRate == "32k" || !readyForConfig) return;
                                    webInfoState.setVoiceChangerConfig(webInfoState.voiceChangerConfig.config.voiceChangerType, "32k", webInfoState.voiceChangerConfig.useF0, webInfoState.voiceChangerConfig.inputLength);
                                }}
                            >
                                32k
                            </span>
                            <span
                                className={!readyForConfig ? "character-area-control-button-disable" : sr40KClassName}
                                onClick={() => {
                                    if (webInfoState.voiceChangerConfig.sampleRate == "40k" || !readyForConfig) return;
                                    webInfoState.setVoiceChangerConfig(webInfoState.voiceChangerConfig.config.voiceChangerType, "40k", webInfoState.voiceChangerConfig.useF0, webInfoState.voiceChangerConfig.inputLength);
                                }}
                            >
                                40k
                            </span>
                        </span>
                    </div>
                </div>
            </div>
        );

        const pitchEnableClassName = "character-area-control-button" + (webInfoState.voiceChangerConfig.useF0 == true ? " character-area-control-button-active" : " character-area-control-button-stanby");
        const pitchDisableClassName = "character-area-control-button" + (webInfoState.voiceChangerConfig.useF0 == false ? " character-area-control-button-active" : " character-area-control-button-stanby");
        const pitch = (
            <div className="character-area-control">
                <div className="character-area-control-title">Pitch</div>
                <div className="character-area-control-field">
                    <div className="character-area-slider-control">
                        <span className="character-area-slider-control-kind"></span>
                        <span className="character-area-control-buttons">
                            <span
                                className={!readyForConfig ? "character-area-control-button-disable" : pitchEnableClassName}
                                onClick={() => {
                                    if (webInfoState.voiceChangerConfig.useF0 == true || !readyForConfig) return;
                                    webInfoState.setVoiceChangerConfig(webInfoState.voiceChangerConfig.config.voiceChangerType, webInfoState.voiceChangerConfig.sampleRate, true, webInfoState.voiceChangerConfig.inputLength);
                                }}
                            >
                                Enable
                            </span>
                            <span
                                className={!readyForConfig ? "character-area-control-button-disable" : pitchDisableClassName}
                                onClick={() => {
                                    if (webInfoState.voiceChangerConfig.useF0 == false || !readyForConfig) return;
                                    webInfoState.setVoiceChangerConfig(webInfoState.voiceChangerConfig.config.voiceChangerType, webInfoState.voiceChangerConfig.sampleRate, false, webInfoState.voiceChangerConfig.inputLength);
                                }}
                            >
                                Disable
                            </span>
                        </span>
                    </div>
                </div>
            </div>
        );

        const latencyHighClassName = "character-area-control-button" + (webInfoState.voiceChangerConfig.inputLength == "24000" ? " character-area-control-button-active" : " character-area-control-button-stanby");
        const latencyMidClassName = "character-area-control-button" + (webInfoState.voiceChangerConfig.inputLength == "12000" ? " character-area-control-button-active" : " character-area-control-button-stanby");
        const latencyLowClassName = "character-area-control-button" + (webInfoState.voiceChangerConfig.inputLength == "8000" ? " character-area-control-button-active" : " character-area-control-button-stanby");
        const latency = (
            <div className="character-area-control">
                <div className="character-area-control-title">Latency</div>
                <div className="character-area-control-field">
                    <div className="character-area-slider-control">
                        <span className="character-area-slider-control-kind"></span>
                        <span className="character-area-control-buttons">
                            <span
                                className={!readyForConfig ? "character-area-control-button-disable" : latencyHighClassName}
                                onClick={() => {
                                    if (webInfoState.voiceChangerConfig.inputLength == "24000" || !readyForConfig) return;
                                    webInfoState.setVoiceChangerConfig(webInfoState.voiceChangerConfig.config.voiceChangerType, webInfoState.voiceChangerConfig.sampleRate, webInfoState.voiceChangerConfig.useF0, "24000");
                                }}
                            >
                                High
                            </span>
                            <span
                                className={!readyForConfig ? "character-area-control-button-disable" : latencyMidClassName}
                                onClick={() => {
                                    if (webInfoState.voiceChangerConfig.inputLength == "12000" || !readyForConfig) return;
                                    webInfoState.setVoiceChangerConfig(webInfoState.voiceChangerConfig.config.voiceChangerType, webInfoState.voiceChangerConfig.sampleRate, webInfoState.voiceChangerConfig.useF0, "12000");
                                }}
                            >
                                Mid
                            </span>
                            <span
                                className={!readyForConfig ? "character-area-control-button-disable" : latencyLowClassName}
                                onClick={() => {
                                    if (webInfoState.voiceChangerConfig.inputLength == "8000" || !readyForConfig) return;
                                    webInfoState.setVoiceChangerConfig(webInfoState.voiceChangerConfig.config.voiceChangerType, webInfoState.voiceChangerConfig.sampleRate, webInfoState.voiceChangerConfig.useF0, "8000");
                                }}
                            >
                                Low
                            </span>
                        </span>
                    </div>
                </div>
            </div>
        );
        return (
            <>
                {verison}
                {sampleRate}
                {pitch}
                {latency}
            </>
        );
    }, [
        serverSetting.serverSetting,
        serverSetting.updateServerSettings,
        selected,
        webInfoState.upkey,
        webInfoState.voiceChangerConfig.config.voiceChangerType,
        webInfoState.voiceChangerConfig.sampleRate,
        webInfoState.voiceChangerConfig.useF0,
        webInfoState.voiceChangerConfig.inputLength,
        webInfoState.webModelLoadingState,
        guiState.isConverting,
        webInfoState.webModelLoadingState,
    ]);

    return settingArea;
};
