import React, { useMemo, useState } from "react";
import { useAppState } from "../../../001_provider/001_AppStateProvider";
import { useGuiState } from "../001_GuiStateProvider";
import { useMessageBuilder } from "../../../hooks/useMessageBuilder";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { useAppRoot } from "../../../001_provider/001_AppRootProvider";
import { DDSPSVCModelSlot } from "@dannadori/voice-changer-client-js";

export type ModelSlotAreaProps = {};

const SortTypes = {
    slot: "slot",
    name: "name",
} as const;
export type SortTypes = (typeof SortTypes)[keyof typeof SortTypes];

export const ModelSlotArea = (_props: ModelSlotAreaProps) => {
    const { serverSetting, getInfo } = useAppState();
    const { appGuiSettingState } = useAppRoot();
    const guiState = useGuiState();
    const messageBuilderState = useMessageBuilder();
    const [sortType, setSortType] = useState<SortTypes>("slot");

    useMemo(() => {
        messageBuilderState.setMessage(__filename, "edit", { ja: "編集", en: "edit" });
    }, []);

    const modelTiles = useMemo(() => {
        if (!serverSetting.serverSetting.modelSlots) {
            return [];
        }
        const modelSlots =
            sortType == "slot"
                ? serverSetting.serverSetting.modelSlots
                : serverSetting.serverSetting.modelSlots.slice().sort((a, b) => {
                    return a.name.localeCompare(b.name);
                });

        return modelSlots
            .map((x, index) => {
                if (!x.modelFile || x.modelFile.length == 0) {
                    return null;
                }
                const tileContainerClass = x.slotIndex == serverSetting.serverSetting.modelSlotIndex ? "model-slot-tile-container-selected" : "model-slot-tile-container";
                const name = x.name.length > 8 ? x.name.substring(0, 7) + "..." : x.name;

                const modelDir = x.slotIndex == "Beatrice-JVS" ? "model_dir_static" : serverSetting.serverSetting.voiceChangerParams.model_dir;
                const icon = x.iconFile.length > 0 ? modelDir + "/" + x.slotIndex + "/" + x.iconFile.split(/[\/\\]/).pop() : "./assets/icons/human.png";

                const iconElem =
                    x.iconFile.length > 0 ? (
                        <>
                            {/* <img className="model-slot-tile-icon" src={serverSetting.serverSetting.voiceChangerParams.model_dir + "/" + x.slotIndex + "/" + x.iconFile.split(/[\/\\]/).pop()} alt={x.name} /> */}
                            <img className="model-slot-tile-icon" src={icon} alt={x.name} />
                            <div className="model-slot-tile-vctype">{x.voiceChangerType}</div>
                        </>
                    ) : (
                        <>
                            <div className="model-slot-tile-icon-no-entry">no image</div>
                            <div className="model-slot-tile-vctype">{x.voiceChangerType}</div>
                        </>
                    );

                const clickAction = async () => {
                    // @ts-ignore
                    const dummyModelSlotIndex = Math.floor(Date.now() / 1000) * 1000 + x.slotIndex;
                    await serverSetting.updateServerSettings({ ...serverSetting.serverSetting, modelSlotIndex: dummyModelSlotIndex });
                    setTimeout(() => {
                        // quick hack
                        getInfo();
                    }, 1000 * 2);
                };

                return (
                    <div key={index} className={tileContainerClass} onClick={clickAction}>
                        <div className="model-slot-tile-icon-div">{iconElem}</div>
                        <div className="model-slot-tile-dscription">{name}</div>
                    </div>
                );
            })
            .filter((x) => x != null);
    }, [serverSetting.serverSetting.modelSlots, serverSetting.serverSetting.modelSlotIndex, sortType]);

    const modelSlotArea = useMemo(() => {
        const onModelSlotEditClicked = () => {
            guiState.stateControls.showModelSlotManagerCheckbox.updateState(true);
        };
        const sortSlotByIdClass = sortType == "slot" ? "model-slot-sort-button-active" : "model-slot-sort-button";
        const sortSlotByNameClass = sortType == "name" ? "model-slot-sort-button-active" : "model-slot-sort-button";
        return (
            <div className="model-slot-area">
                <div className="model-slot-panel">
                    <div className="model-slot-tiles-container">{modelTiles}</div>
                    <div className="model-slot-buttons">
                        <div className="model-slot-sort-buttons">
                            <div
                                className={sortSlotByIdClass}
                                onClick={() => {
                                    setSortType("slot");
                                }}
                            >
                                <FontAwesomeIcon icon={["fas", "arrow-down-1-9"]} style={{ fontSize: "1rem" }} />
                            </div>
                            <div
                                className={sortSlotByNameClass}
                                onClick={() => {
                                    setSortType("name");
                                }}
                            >
                                <FontAwesomeIcon icon={["fas", "arrow-down-a-z"]} style={{ fontSize: "1rem" }} />
                            </div>
                        </div>
                        <div className="model-slot-button" onClick={onModelSlotEditClicked}>
                            {messageBuilderState.getMessage(__filename, "edit")}
                        </div>
                    </div>
                </div>
            </div>
        );
    }, [modelTiles, sortType]);

    if (appGuiSettingState.edition.indexOf("web") >= 0) {
        return <></>;
    }

    return modelSlotArea;
};
