import React, { useMemo } from "react";
import { useAppState } from "../../../001_provider/001_AppStateProvider";

export type SoVitsSVC40SettingAreaProps = {};

export const SoVitsSVC40SettingArea = (_props: SoVitsSVC40SettingAreaProps) => {
    const { serverSetting } = useAppState();

    const selected = useMemo(() => {
        if (serverSetting.serverSetting.modelSlotIndex == undefined) {
            return;
        } else if (serverSetting.serverSetting.modelSlotIndex == "Beatrice-JVS") {
            const beatriceJVS = serverSetting.serverSetting.modelSlots.find((v) => v.slotIndex == "Beatrice-JVS");
            return beatriceJVS;
        } else {
            return serverSetting.serverSetting.modelSlots[serverSetting.serverSetting.modelSlotIndex];
        }
    }, [serverSetting.serverSetting.modelSlotIndex, serverSetting.serverSetting.modelSlots]);

    const settingArea = useMemo(() => {
        if (!selected) {
            return <></>;
        }

        if (selected.voiceChangerType != "so-vits-svc-40") {
            return <></>;
        }

        const cluster = (
            <div className="character-area-control">
                <div className="character-area-control-title">Cluster:</div>
                <div className="character-area-control-field">
                    <div className="character-area-slider-control">
                        <span className="character-area-slider-control-kind"></span>
                        <span className="character-area-slider-control-slider">
                            <input
                                type="range"
                                min="0"
                                max="1.0"
                                step="0.1"
                                value={serverSetting.serverSetting.clusterInferRatio}
                                onChange={(e) => {
                                    serverSetting.updateServerSettings({ ...serverSetting.serverSetting, clusterInferRatio: Number(e.target.value) });
                                }}
                            ></input>
                        </span>
                        <span className="character-area-slider-control-val">{serverSetting.serverSetting.clusterInferRatio}</span>
                    </div>
                </div>
            </div>
        );

        const noise = (
            <div className="character-area-control">
                <div className="character-area-control-title">Noise:</div>
                <div className="character-area-control-field">
                    <div className="character-area-slider-control">
                        <span className="character-area-slider-control-kind"></span>
                        <span className="character-area-slider-control-slider">
                            <input
                                type="range"
                                min="0"
                                max="1.0"
                                step="0.1"
                                value={serverSetting.serverSetting.noiseScale}
                                onChange={(e) => {
                                    serverSetting.updateServerSettings({ ...serverSetting.serverSetting, noiseScale: Number(e.target.value) });
                                }}
                            ></input>
                        </span>
                        <span className="character-area-slider-control-val">{serverSetting.serverSetting.noiseScale}</span>
                    </div>
                </div>
            </div>
        );

        return (
            <>
                {cluster}
                {noise}
            </>
        );
    }, [serverSetting.serverSetting, serverSetting.updateServerSettings, selected]);

    return settingArea;
};
