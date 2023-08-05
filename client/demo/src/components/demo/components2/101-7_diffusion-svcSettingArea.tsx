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

        const skipDiffusionClass = serverSetting.serverSetting.skipDiffusion == 0 ? "character-area-toggle-button" : "character-area-toggle-button-active";

        const skipDiffRow = (
            <div className="character-area-control">
                <div className="character-area-control-title">Boost</div>
                <div className="character-area-control-field">
                    <div className="character-area-buttons">
                        <div
                            className={skipDiffusionClass}
                            onClick={() => {
                                serverSetting.updateServerSettings({ ...serverSetting.serverSetting, skipDiffusion: serverSetting.serverSetting.skipDiffusion == 0 ? 1 : 0 });
                            }}
                        >
                            skip diff
                        </div>
                    </div>
                </div>
            </div>
        );

        const skipValues = getDivisors(serverSetting.serverSetting.kStep);
        skipValues.pop();

        const kStepRow = (
            <div className="character-area-control">
                <div className="character-area-control-title">k-step:</div>
                <div className="character-area-control-field">
                    <div className="character-area-slider-control">
                        <span className="character-area-slider-control-kind"></span>
                        <span className="character-area-slider-control-slider">
                            <input
                                type="range"
                                min="2"
                                max={(selected as DiffusionSVCModelSlot).kStepMax}
                                step="1"
                                value={serverSetting.serverSetting.kStep}
                                onChange={(e) => {
                                    const newKStep = Number(e.target.value);
                                    const newSkipValues = getDivisors(Number(e.target.value));
                                    newSkipValues.pop();
                                    serverSetting.updateServerSettings({ ...serverSetting.serverSetting, speedUp: Math.max(...newSkipValues), kStep: newKStep });
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
                <div className="character-area-control-title">skip</div>
                <div className="character-area-control-field">
                    <div className="character-area-slider-control">
                        <span className="character-area-slider-control-kind"></span>
                        <span className="character-area-slider-control-slider">
                            <select
                                name=""
                                id=""
                                value={serverSetting.serverSetting.speedUp}
                                onChange={(e) => {
                                    serverSetting.updateServerSettings({ ...serverSetting.serverSetting, speedUp: Number(e.target.value) });
                                }}
                            >
                                {skipValues.map((v) => {
                                    return (
                                        <option value={v} key={v}>
                                            {v}
                                        </option>
                                    );
                                })}
                            </select>
                        </span>
                    </div>
                </div>
            </div>
        );
        return (
            <>
                {skipDiffRow}
                {kStepRow}
                {speedUpRow}
            </>
        );
    }, [serverSetting.serverSetting, serverSetting.updateServerSettings, selected]);

    return settingArea;
};

const getDivisors = (num: number) => {
    var divisors = [];
    var end = Math.sqrt(num);

    for (var i = 1; i <= end; i++) {
        if (num % i === 0) {
            divisors.push(i);
            if (i !== num / i) {
                divisors.push(num / i);
            }
        }
    }

    return divisors.sort((a, b) => a - b);
};
