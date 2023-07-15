import React, { useMemo } from "react";
import { useAppState } from "../../../001_provider/001_AppStateProvider";
import { DiffusionSVCModelSlot } from "@dannadori/voice-changer-client-js";

export type DiffusionSVCSettingAreaProps = {};

export const DiffusionSVCSettingArea = (_props: DiffusionSVCSettingAreaProps) => {
    const { serverSetting } = useAppState();

    const selected = useMemo(() => {
        if (serverSetting.serverSetting.modelSlotIndex == undefined) {
            return;
        }
        return serverSetting.serverSetting.modelSlots[serverSetting.serverSetting.modelSlotIndex];
    }, [serverSetting.serverSetting.modelSlotIndex, serverSetting.serverSetting.modelSlots]);

    const settingArea = useMemo(() => {
        if (!selected) {
            return <></>;
        }

        if (selected.voiceChangerType != "Diffusion-SVC") {
            return <></>;
        }

        const kStepRow = (
            <div className="character-area-control">
                <div className="character-area-control-title">k-step:</div>
                <div className="character-area-control-field">
                    <div className="character-area-slider-control">
                        <span className="character-area-slider-control-kind"></span>
                        <span className="character-area-slider-control-slider">
                            <input
                                type="range"
                                min="0"
                                max={(selected as DiffusionSVCModelSlot).kStepMax}
                                step="1"
                                value={serverSetting.serverSetting.kStep}
                                onChange={(e) => {
                                    serverSetting.updateServerSettings({ ...serverSetting.serverSetting, kStep: Number(e.target.value) });
                                }}
                            ></input>
                        </span>
                        <span className="character-area-slider-control-val">{serverSetting.serverSetting.kStep}</span>
                    </div>
                </div>
            </div>
        );
        const speedUpRow = (
            <div className="character-area-control">
                <div className="character-area-control-title">speedup</div>
                <div className="character-area-control-field">
                    <div className="character-area-slider-control">
                        <span className="character-area-slider-control-kind"></span>
                        <span className="character-area-slider-control-slider">
                            <input
                                type="range"
                                min="0"
                                max={serverSetting.serverSetting.kStep}
                                step="1"
                                value={serverSetting.serverSetting.speedUp}
                                onChange={(e) => {
                                    serverSetting.updateServerSettings({ ...serverSetting.serverSetting, speedUp: Number(e.target.value) });
                                }}
                            ></input>
                        </span>
                        <span className="character-area-slider-control-val">{serverSetting.serverSetting.speedUp}</span>
                    </div>
                </div>
            </div>
        );
        return (
            <>
                {kStepRow}
                {speedUpRow}
            </>
        );
    }, [serverSetting.serverSetting, serverSetting.updateServerSettings, selected]);

    return settingArea;
};
