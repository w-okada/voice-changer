import React, { useMemo } from "react";
import { useAppState } from "../../../001_provider/001_AppStateProvider";
import { F0Detector } from "@dannadori/voice-changer-client-js";
import { useAppRoot } from "../../../001_provider/001_AppRootProvider";

export type QualityAreaProps = {
    detectors: string[];
};

export const QualityArea = (props: QualityAreaProps) => {
    const { setVoiceChangerClientSetting, serverSetting, setting } = useAppState();
    const { appGuiSettingState } = useAppRoot();
    const edition = appGuiSettingState.edition;
    const webEdition = appGuiSettingState.edition.indexOf("web") >= 0;

    const qualityArea = useMemo(() => {
        if (!serverSetting.updateServerSettings || !setVoiceChangerClientSetting || !serverSetting.serverSetting || !setting) {
            return <></>;
        }

        const generateF0DetOptions = () => {
            if (edition.indexOf("onnxdirectML-cuda") >= 0) {
                const recommended = ["crepe_tiny", "rmvpe_onnx"];
                return Object.values(props.detectors).map((x) => {
                    if (recommended.includes(x)) {
                        return (
                            <option key={x} value={x}>
                                {x}
                            </option>
                        );
                    } else {
                        return (
                            <option key={x} value={x} disabled>
                                {x}(N/A)
                            </option>
                        );
                    }
                });
            } else {
                return Object.values(props.detectors).map((x) => {
                    return (
                        <option key={x} value={x}>
                            {x}
                        </option>
                    );
                });
            }
        };
        const f0DetOptions = generateF0DetOptions();

        const f0Det = webEdition ? (
            <></>
        ) : (
            <div className="config-sub-area-control">
                <div className="config-sub-area-control-title">F0 Det.:</div>
                <div className="config-sub-area-control-field">
                    <select
                        className="body-select"
                        value={serverSetting.serverSetting.f0Detector}
                        onChange={(e) => {
                            serverSetting.updateServerSettings({ ...serverSetting.serverSetting, f0Detector: e.target.value as F0Detector });
                        }}
                    >
                        {f0DetOptions}
                    </select>
                </div>
            </div>
        );

        const threshold = webEdition ? (
            <></>
        ) : (
            <div className="config-sub-area-control">
                <div className="config-sub-area-control-title">S.Thresh.:</div>
                <div className="config-sub-area-control-field">
                    <div className="config-sub-area-slider-control">
                        <span className="config-sub-area-slider-control-kind"></span>
                        <span className="config-sub-area-slider-control-slider">
                            <input
                                type="range"
                                className="config-sub-area-slider-control-slider"
                                min="0.00000"
                                max="0.001"
                                step="0.00001"
                                value={serverSetting.serverSetting.silentThreshold || 0}
                                onChange={(e) => {
                                    serverSetting.updateServerSettings({ ...serverSetting.serverSetting, silentThreshold: Number(e.target.value) });
                                }}
                            ></input>
                        </span>
                        <span className="config-sub-area-slider-control-val">{serverSetting.serverSetting.silentThreshold}</span>
                    </div>
                </div>
            </div>
        );

        return (
            <div className="config-sub-area">
                <div className="config-sub-area-control">
                    <div className="config-sub-area-control-title">NOISE:</div>
                    <div className="config-sub-area-control-field">
                        <div className="config-sub-area-noise-container">
                            <div className="config-sub-area-noise-checkbox-container">
                                <input
                                    type="checkbox"
                                    disabled={serverSetting.serverSetting.enableServerAudio != 0}
                                    checked={setting.voiceChangerClientSetting.echoCancel}
                                    onChange={(e) => {
                                        try {
                                            setVoiceChangerClientSetting({ ...setting.voiceChangerClientSetting, echoCancel: e.target.checked });
                                        } catch (e) {
                                            console.error(e);
                                        }
                                    }}
                                />{" "}
                                <span>Echo</span>
                            </div>
                            <div className="config-sub-area-noise-checkbox-container">
                                <input
                                    type="checkbox"
                                    disabled={serverSetting.serverSetting.enableServerAudio != 0}
                                    checked={setting.voiceChangerClientSetting.noiseSuppression}
                                    onChange={(e) => {
                                        try {
                                            setVoiceChangerClientSetting({ ...setting.voiceChangerClientSetting, noiseSuppression: e.target.checked });
                                        } catch (e) {
                                            console.error(e);
                                        }
                                    }}
                                />{" "}
                                <span>Sup1</span>
                            </div>
                            <div className="config-sub-area-noise-checkbox-container">
                                <input
                                    type="checkbox"
                                    disabled={serverSetting.serverSetting.enableServerAudio != 0}
                                    checked={setting.voiceChangerClientSetting.noiseSuppression2}
                                    onChange={(e) => {
                                        try {
                                            setVoiceChangerClientSetting({ ...setting.voiceChangerClientSetting, noiseSuppression2: e.target.checked });
                                        } catch (e) {
                                            console.error(e);
                                        }
                                    }}
                                />{" "}
                                <span>Sup2</span>
                            </div>
                        </div>
                    </div>
                </div>
                {f0Det}
                {threshold}
            </div>
        );
    }, [serverSetting.serverSetting, setting, serverSetting.updateServerSettings, setVoiceChangerClientSetting]);

    return qualityArea;
};
