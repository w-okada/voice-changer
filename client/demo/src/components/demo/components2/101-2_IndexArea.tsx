import React, { useMemo } from "react";
import { useAppState } from "../../../001_provider/001_AppStateProvider";

export type IndexAreaProps = {};

export const IndexArea = (_props: IndexAreaProps) => {
    const { serverSetting } = useAppState();

    const selected = useMemo(() => {
        if (serverSetting.serverSetting.modelSlotIndex == undefined) {
            return;
        } else {
            return serverSetting.serverSetting.modelSlots[serverSetting.serverSetting.modelSlotIndex];
        }
    }, [serverSetting.serverSetting.modelSlotIndex, serverSetting.serverSetting.modelSlots]);

    const indexArea = useMemo(() => {
        if (!selected) {
            return <></>;
        }
        if (selected.voiceChangerType != "RVC") {
            return <></>;
        }

        const currentIndexRatio = serverSetting.serverSetting.indexRatio;
        const indexRatioValueUpdatedAction = async (val: number) => {
            await serverSetting.updateServerSettings({ ...serverSetting.serverSetting, indexRatio: val });
        };

        return (
            <div className="character-area-control">
                <div className="character-area-control-title">INDEX:</div>
                <div className="character-area-control-field">
                    <div className="character-area-slider-control">
                        <span className="character-area-slider-control-kind"></span>
                        <span className="character-area-slider-control-slider">
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.1"
                                value={currentIndexRatio}
                                onChange={(e) => {
                                    indexRatioValueUpdatedAction(Number(e.target.value));
                                }}
                            ></input>
                        </span>
                        <span className="character-area-slider-control-val">{currentIndexRatio}</span>
                    </div>
                </div>
            </div>
        );
    }, [serverSetting.serverSetting, serverSetting.updateServerSettings, selected]);

    return indexArea;
};
