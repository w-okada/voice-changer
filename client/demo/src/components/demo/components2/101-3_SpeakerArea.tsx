import React, { useMemo } from "react";
import { useAppState } from "../../../001_provider/001_AppStateProvider";

export type SpeakerAreaProps = {};

export const SpeakerArea = (_props: SpeakerAreaProps) => {
    const { serverSetting } = useAppState();

    const selected = useMemo(() => {
        if (serverSetting.serverSetting.modelSlotIndex == undefined) {
            return;
        } else {
            return serverSetting.serverSetting.modelSlots[serverSetting.serverSetting.modelSlotIndex];
        }
    }, [serverSetting.serverSetting.modelSlotIndex, serverSetting.serverSetting.modelSlots]);

    const dstArea = useMemo(() => {
        if (!selected) {
            return <></>;
        }

        const options = Object.keys(selected.speakers).map((key) => {
            const val = selected.speakers[Number(key)];
            return (
                <option key={key} value={key}>
                    {val}[{key}]
                </option>
            );
        });

        const srcSpeakerValueUpdatedAction = async (val: number) => {
            await serverSetting.updateServerSettings({ ...serverSetting.serverSetting, dstId: val });
        };

        return (
            <div className="character-area-control">
                <div className="character-area-control-title">{selected.voiceChangerType == "RVC" ? "Voice:" : ""}</div>
                <div className="character-area-control-field">
                    <div className="character-area-slider-control">
                        <span className="character-area-slider-control-kind"></span>
                        <span className="character-area-slider-control-slider">
                            <select
                                value={serverSetting.serverSetting.dstId}
                                onChange={(e) => {
                                    srcSpeakerValueUpdatedAction(Number(e.target.value));
                                }}
                            >
                                {options}
                            </select>
                        </span>
                    </div>
                </div>
            </div>
        );
    }, [serverSetting.serverSetting, serverSetting.updateServerSettings, selected]);

    return (
        <>
            {dstArea}
        </>
    );
};
