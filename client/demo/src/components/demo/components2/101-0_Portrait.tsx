import React, { useEffect, useMemo, useState } from "react";
import { useAppState } from "../../../001_provider/001_AppStateProvider";
import { useMessageBuilder } from "../../../hooks/useMessageBuilder";
import { useAppRoot } from "../../../001_provider/001_AppRootProvider";
export type PortraitProps = {};
const BeatriceSpeakerType = {
    male: "male",
    female: "female",
} as const;
type BeatriceSpeakerType = (typeof BeatriceSpeakerType)[keyof typeof BeatriceSpeakerType];

// @ts-ignore
import MyIcon from "./female-clickable.svg";
import { useGuiState } from "../001_GuiStateProvider";
export const Portrait = (_props: PortraitProps) => {
    const { appGuiSettingState } = useAppRoot();
    const { serverSetting, volume, bufferingTime, performance, webInfoState } = useAppState();
    const messageBuilderState = useMessageBuilder();
    const [beatriceSpeakerType, setBeatriceSpeakerType] = useState<BeatriceSpeakerType>(BeatriceSpeakerType.male);
    const [beatriceSpeakerIndexInGender, setBeatriceSpeakerIndexInGender] = useState<string>("");
    const { setBeatriceJVSSpeakerId } = useGuiState();

    const beatriceMaleSpeakersList = [1, 3, 5, 6, 9, 11, 12, 13, 20, 21, 22, 23, 28, 31, 32, 33, 34, 37, 41, 42, 44, 45, 46, 47, 48, 49, 50, 52, 54, 68, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81, 86, 87, 88, 89, 97, 98, 99, 100];
    const beatriceFemaleSpeakersList = [2, 4, 7, 8, 10, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27, 29, 30, 35, 36, 38, 39, 40, 43, 51, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 69, 72, 82, 83, 84, 85, 90, 91, 92, 93, 94, 95, 96];

    const webEdition = appGuiSettingState.edition.indexOf("web") >= 0;

    useMemo(() => {
        messageBuilderState.setMessage(__filename, "terms_of_use", { ja: "利用規約", en: "terms of use" });
    }, []);

    const selected = useMemo(() => {
        if (webEdition) {
            return webInfoState.webModelslot;
        }

        if (serverSetting.serverSetting.modelSlotIndex == undefined) {
            return;
        } else if (serverSetting.serverSetting.modelSlotIndex == "Beatrice-JVS") {
            const beatriceJVS = serverSetting.serverSetting.modelSlots.find((v) => v.slotIndex == "Beatrice-JVS");
            return beatriceJVS;
        } else {
            return serverSetting.serverSetting.modelSlots[serverSetting.serverSetting.modelSlotIndex];
        }
    }, [serverSetting.serverSetting.modelSlotIndex, serverSetting.serverSetting.modelSlots, webEdition]);

    useEffect(() => {
        const vol = document.getElementById("status-vol") as HTMLSpanElement;
        const buf = document.getElementById("status-buf") as HTMLSpanElement;
        const res = document.getElementById("status-res") as HTMLSpanElement;
        if (!vol || !buf || !res) {
            return;
        }
        vol.innerText = volume.toFixed(4);
        buf.innerText = bufferingTime.toString();
        res.innerText = performance.responseTime.toString();
    }, [volume, bufferingTime, performance]);

    const setSelectedClass = () => {
        const iframe = document.querySelector(".beatrice-speaker-graph-container");
        if (!iframe) {
            return;
        }
        // @ts-ignore
        const svgDoc = iframe.contentDocument;
        const gElements = svgDoc.getElementsByClassName("beatrice-node-pointer");
        for (const gElement of gElements) {
            gElement.classList.remove("beatrice-node-pointer-selected");
        }
        const keys = beatriceSpeakerIndexInGender.split("-");
        const id = keys.pop();
        const gender = keys.pop();

        if (beatriceSpeakerType == gender) {
            const selected = svgDoc.getElementById(`beatrice-node-${gender}-${id}`);
            selected?.classList.add("beatrice-node-pointer-selected");
        }
    };

    const setBeatriceSpeakerIndex = async (elementId: string) => {
        setBeatriceSpeakerIndexInGender(elementId);
        const keys = elementId.split("-");
        const id = Number(keys.pop());
        const gender = keys.pop();
        let beatriceSpeakerIndex;
        if (gender == "male") {
            beatriceSpeakerIndex = beatriceMaleSpeakersList[id];
        } else {
            beatriceSpeakerIndex = beatriceFemaleSpeakersList[id];
        }
        setBeatriceJVSSpeakerId(beatriceSpeakerIndex);
    };

    useEffect(() => {
        const iframe = document.querySelector(".beatrice-speaker-graph-container");
        if (!iframe) {
            return;
        }
        const setOnClick = () => {
            // @ts-ignore
            const svgDoc = iframe.contentDocument;

            const gElements = svgDoc.getElementsByClassName("beatrice-node-pointer");
            const textElements = svgDoc.getElementsByClassName("beatrice-text-pointer");
            for (const gElement of gElements) {
                gElement.onclick = () => {
                    setBeatriceSpeakerIndex(gElement.id);
                };
            }
            for (const textElement of textElements) {
                textElement.onclick = () => {
                    setBeatriceSpeakerIndex(textElement.id);
                };
            }
            setSelectedClass();
        };
        iframe.addEventListener("load", setOnClick);
        return () => {
            iframe.removeEventListener("load", setOnClick);
        };
    }, [selected, beatriceSpeakerType]);

    useEffect(() => {
        setSelectedClass();
    }, [selected, beatriceSpeakerType, beatriceSpeakerIndexInGender]);

    const portrait = useMemo(() => {
        if (!selected) {
            return <></>;
        }

        let portrait;
        if (webEdition) {
            const icon = selected.iconFile;
            portrait = <img className="portrait" src={icon} alt={selected.name} />;
        } else if (selected.slotIndex == "Beatrice-JVS") {
            const maleButtonClass = beatriceSpeakerType == "male" ? "button-selected" : "button";
            const femaleButtonClass = beatriceSpeakerType == "male" ? "button" : "button-selected";
            const svgURL = beatriceSpeakerType == "male" ? "./assets/beatrice/male-clickable.svg" : "./assets/beatrice/female-clickable.svg";
            portrait = (
                <>
                    <div className="beatrice-portrait-title">
                        Beatrice <span className="edition">JVS Corpus</span>
                    </div>
                    <div className="beatrice-portrait-select">
                        <div
                            className={maleButtonClass}
                            onClick={() => {
                                setBeatriceSpeakerType(BeatriceSpeakerType.male);
                            }}
                        >
                            male
                        </div>
                        <div
                            className={femaleButtonClass}
                            onClick={() => {
                                setBeatriceSpeakerType(BeatriceSpeakerType.female);
                            }}
                        >
                            female
                        </div>
                    </div>
                    {/* <iframe className="beatrice-speaker-graph-container" style={{ width: "20rem", height: "20rem", border: "none" }} src="./assets/beatrice/female-clickable.svg" title="terms_of_use" sandbox="allow-same-origin allow-scripts allow-popups allow-forms"></iframe> */}
                    <iframe className="beatrice-speaker-graph-container" src={svgURL} title="beatrice JVS Corpus speakers" sandbox="allow-same-origin allow-scripts allow-popups allow-forms"></iframe>
                </>
            );
        } else {
            const modelDir = serverSetting.serverSetting.modelSlotIndex == "Beatrice-JVS" ? "model_dir_static" : serverSetting.serverSetting.voiceChangerParams.model_dir;
            const icon = selected.iconFile.length > 0 ? modelDir + "/" + selected.slotIndex + "/" + selected.iconFile.split(/[\/\\]/).pop() : "./assets/icons/human.png";
            portrait = <img className="portrait" src={icon} alt={selected.name} />;
        }
        const selectedTermOfUseUrlLink = selected.termsOfUseUrl ? (
            <a href={selected.termsOfUseUrl} target="_blank" rel="noopener noreferrer" className="portrait-area-terms-of-use-link">
                [{messageBuilderState.getMessage(__filename, "terms_of_use")}]
            </a>
        ) : (
            <></>
        );

        return (
            <div className="portrait-area">
                <div className="portrait-container">
                    {portrait}
                    <div className="portrait-area-status">
                        <p>
                            <span className="portrait-area-status-vctype">{selected.voiceChangerType}</span>
                        </p>
                        <p>
                            vol: <span id="status-vol">0</span>
                        </p>
                        <p>
                            buf: <span id="status-buf">0</span> ms
                        </p>
                        <p>
                            res: <span id="status-res">0</span> ms
                        </p>
                    </div>
                    <div className="portrait-area-terms-of-use">{selectedTermOfUseUrlLink}</div>
                </div>
            </div>
        );
    }, [selected, beatriceSpeakerType]);

    return portrait;
};
