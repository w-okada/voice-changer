import React, { useMemo } from "react";
import { useAppState } from "../../../001_provider/001_AppStateProvider";
import { useGuiState } from "../001_GuiStateProvider";

export type TuningAreaProps = {};

export const TuningArea = (_props: TuningAreaProps) => {
    const { serverSetting, webInfoState, webEdition } = useAppState();

    const selected = useMemo(() => {
        if (webEdition) {
            return webInfoState.webModelslot;
        }
        if (serverSetting.serverSetting.modelSlotIndex == undefined) {
            return;
        } else {
            return serverSetting.serverSetting.modelSlots[serverSetting.serverSetting.modelSlotIndex];
        }
    }, [serverSetting.serverSetting.modelSlotIndex, serverSetting.serverSetting.modelSlots, webEdition]);

    const tuningArea = useMemo(() => {
        if (!selected) {
            return <></>;
        }

        let currentTuning;
        if (webEdition) {
            currentTuning = webInfoState.upkey;
        } else {
            currentTuning = serverSetting.serverSetting.tran;
        }
        const tranValueUpdatedAction = async (val: number) => {
            if (webEdition) {
                webInfoState.setUpkey(val);
            } else {
                await serverSetting.updateServerSettings({ ...serverSetting.serverSetting, tran: val });
            }
        };

        return (
            <div className="character-area-control">
                <div className="character-area-control-title">TUNE:</div>
                <div className="character-area-control-field">
                    <div className="character-area-slider-control">
                        <span className="character-area-slider-control-kind"></span>
                        <span className="character-area-slider-control-slider">
                            <input
                                type="range"
                                min="-50"
                                max="50"
                                step="1"
                                value={currentTuning}
                                onChange={(e) => {
                                    tranValueUpdatedAction(Number(e.target.value));
                                }}
                            ></input>
                        </span>
                        <span className="character-area-slider-control-val">{currentTuning}</span>
                    </div>
                </div>
            </div>
        );
    }, [serverSetting.serverSetting, serverSetting.updateServerSettings, selected, webEdition, webInfoState.upkey]);

    return tuningArea;
};
