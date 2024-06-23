import React, { useMemo } from "react";
import { useAppState } from "../../../001_provider/001_AppStateProvider";

export type FormantShiftAreaProps = {};

export const FormantShiftArea = (_props: FormantShiftAreaProps) => {
    const { serverSetting } = useAppState();

    const selected = useMemo(() => {
        if (serverSetting.serverSetting.modelSlotIndex == undefined) {
            return;
        } else {
            return serverSetting.serverSetting.modelSlots[serverSetting.serverSetting.modelSlotIndex];
        }
    }, [serverSetting.serverSetting.modelSlotIndex, serverSetting.serverSetting.modelSlots]);

    const formantShiftArea = useMemo(() => {
        if (!selected) {
            return <></>;
        }

        const currentFormantShift = serverSetting.serverSetting.formantShift;
        const formantShiftValueUpdatedAction = async (val: number) => {
            await serverSetting.updateServerSettings({ ...serverSetting.serverSetting, formantShift: val });
        };

        return (
            <div className="character-area-control">
                <div className="character-area-control-title">FORMANT SHIFT:</div>
                <div className="character-area-control-field">
                    <div className="character-area-slider-control">
                        <span className="character-area-slider-control-kind"></span>
                        <span className="character-area-slider-control-slider">
                            <input
                                type="range"
                                min="-5"
                                max="5"
                                step="0.1"
                                value={currentFormantShift}
                                onChange={(e) => {
                                    formantShiftValueUpdatedAction(Number(e.target.value));
                                }}
                            ></input>
                        </span>
                        <span className="character-area-slider-control-val">{currentFormantShift}</span>
                    </div>
                </div>
            </div>
        );
    }, [serverSetting.serverSetting, serverSetting.updateServerSettings, selected]);

    return formantShiftArea;
};
