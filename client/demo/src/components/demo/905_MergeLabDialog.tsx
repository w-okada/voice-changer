import React, { useEffect, useMemo, useState } from "react";
import { useGuiState } from "./001_GuiStateProvider";
import { useAppState } from "../../001_provider/001_AppStateProvider";
import { MergeElement, RVCModelSlot, RVCModelType, VoiceChangerType } from "@dannadori/voice-changer-client-js";

export const MergeLabDialog = () => {
    const guiState = useGuiState();

    const { serverSetting } = useAppState();
    const [currentFilter, setCurrentFilter] = useState<string>("");
    const [mergeElements, setMergeElements] = useState<MergeElement[]>([]);

    // スロットが変更されたときの初期化処理
    const newSlotChangeKey = useMemo(() => {
        if (!serverSetting.serverSetting.modelSlots) {
            return "";
        }
        return serverSetting.serverSetting.modelSlots.reduce((prev, cur) => {
            return prev + "_" + cur.modelFile;
        }, "");
    }, [serverSetting.serverSetting.modelSlots]);

    const filterItems = useMemo(() => {
        return serverSetting.serverSetting.modelSlots.reduce(
            (prev, cur) => {
                if (cur.voiceChangerType != "RVC") {
                    return prev;
                }
                const curRVC = cur as RVCModelSlot;
                const key = `${curRVC.modelType},${cur.samplingRate},${curRVC.embChannels}`;
                const val = { type: curRVC.modelType, samplingRate: cur.samplingRate, embChannels: curRVC.embChannels };
                const existKeys = Object.keys(prev);
                if (!cur.modelFile || cur.modelFile.length == 0) {
                    return prev;
                }
                if (curRVC.modelType == "onnxRVC" || curRVC.modelType == "onnxRVCNono") {
                    return prev;
                }
                if (!existKeys.includes(key)) {
                    prev[key] = val;
                }
                return prev;
            },
            {} as { [key: string]: { type: RVCModelType; samplingRate: number; embChannels: number } },
        );
    }, [newSlotChangeKey]);

    const models = useMemo(() => {
        return serverSetting.serverSetting.modelSlots.filter((x) => {
            if (x.voiceChangerType != "RVC") {
                return;
            }
            const xRVC = x as RVCModelSlot;
            const filterVals = filterItems[currentFilter];
            if (!filterVals) {
                return false;
            }
            if (xRVC.modelType == filterVals.type && xRVC.samplingRate == filterVals.samplingRate && xRVC.embChannels == filterVals.embChannels) {
                return true;
            } else {
                return false;
            }
        });
    }, [filterItems, currentFilter]);

    useEffect(() => {
        if (Object.keys(filterItems).length > 0) {
            setCurrentFilter(Object.keys(filterItems)[0]);
        }
    }, [filterItems]);
    useEffect(() => {
        // models はフィルタ後の配列
        const newMergeElements = models.map((x) => {
            return { slotIndex: x.slotIndex, filename: x.modelFile, strength: 0 };
        });
        setMergeElements(newMergeElements);
    }, [models]);

    const dialog = useMemo(() => {
        const closeButtonRow = (
            <div className="body-row split-3-4-3 left-padding-1">
                <div className="body-item-text"></div>
                <div className="body-button-container body-button-container-space-around">
                    <div
                        className="body-button"
                        onClick={() => {
                            guiState.stateControls.showMergeLabCheckbox.updateState(false);
                        }}
                    >
                        close
                    </div>
                </div>
                <div className="body-item-text"></div>
            </div>
        );

        const filterOptions = Object.keys(filterItems)
            .map((x) => {
                return (
                    <option key={x} value={x}>
                        {x}
                    </option>
                );
            })
            .filter((x) => x != null);

        const onMergeElementsChanged = (slotIndex: number, strength: number) => {
            const newMergeElements = mergeElements.map((x) => {
                if (x.slotIndex == slotIndex) {
                    return { slotIndex: x.slotIndex, filename: x.filename, strength: strength };
                } else {
                    return x;
                }
            });
            setMergeElements(newMergeElements);
        };

        const onMergeClicked = () => {
            const validMergeElements = mergeElements.filter((x) => {
                return x.strength > 0;
            });
            serverSetting.mergeModel({
                voiceChangerType: VoiceChangerType.RVC,
                command: "mix",
                files: validMergeElements,
            });
        };

        const modelList = mergeElements.map((x, index) => {
            const name =
                models.find((model) => {
                    return model.slotIndex == x.slotIndex;
                })?.name || "";

            return (
                <div key={index} className="merge-lab-model-item">
                    <div>{name}</div>
                    <div>
                        <input
                            type="range"
                            className="body-item-input-slider"
                            min="0"
                            max="100"
                            step="1"
                            value={x.strength}
                            onChange={(e) => {
                                onMergeElementsChanged(x.slotIndex, Number(e.target.value));
                            }}
                        ></input>
                        <span className="body-item-input-slider-val">{x.strength}</span>
                    </div>
                </div>
            );
        });

        const content = (
            <div className="merge-lab-container">
                <div className="merge-lab-type-filter">
                    <div>Type:</div>
                    <div>
                        <select
                            value={currentFilter}
                            onChange={(e) => {
                                setCurrentFilter(e.target.value);
                            }}
                        >
                            {filterOptions}
                        </select>
                    </div>
                </div>
                <div className="merge-lab-manipulator">
                    <div className="merge-lab-model-list">{modelList}</div>
                    <div className="merge-lab-merge-buttons">
                        <div className="merge-lab-merge-buttons-notice">The merged model is stored in the final slot. If you assign this slot, it will be overwritten.</div>
                        <div className="merge-lab-merge-button" onClick={onMergeClicked}>
                            merge
                        </div>
                    </div>
                </div>
            </div>
        );
        return (
            <div className="dialog-frame">
                <div className="dialog-title">MergeLab</div>
                <div className="dialog-content">
                    {content}
                    {closeButtonRow}
                </div>
            </div>
        );
    }, [newSlotChangeKey, currentFilter, mergeElements, models]);
    return dialog;
};
