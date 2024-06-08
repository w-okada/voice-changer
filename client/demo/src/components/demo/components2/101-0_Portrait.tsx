import React, { useEffect, useMemo } from "react";
import { useAppState } from "../../../001_provider/001_AppStateProvider";
import { useMessageBuilder } from "../../../hooks/useMessageBuilder";
export type PortraitProps = {};

// @ts-ignore
import MyIcon from "./female-clickable.svg";
export const Portrait = (_props: PortraitProps) => {
    const { serverSetting, volume, bufferingTime, performance } = useAppState();
    const messageBuilderState = useMessageBuilder();

    useMemo(() => {
        messageBuilderState.setMessage(__filename, "terms_of_use", { ja: "利用規約", en: "terms of use" });
    }, []);

    const selected = useMemo(() => {
        if (serverSetting.serverSetting.modelSlotIndex == undefined) {
            return;
        } else {
            return serverSetting.serverSetting.modelSlots[serverSetting.serverSetting.modelSlotIndex];
        }
    }, [serverSetting.serverSetting.modelSlotIndex, serverSetting.serverSetting.modelSlots]);

    useEffect(() => {
        const vol = document.getElementById("status-vol") as HTMLSpanElement;
        const buf = document.getElementById("status-buf") as HTMLSpanElement;
        const res = document.getElementById("status-res") as HTMLSpanElement;
        const perf = document.getElementById("status-perf") as HTMLSpanElement;
        if (!vol || !buf || !res) {
            return;
        }
        vol.innerText = volume.toFixed(4);
        buf.innerText = bufferingTime.toString();
        res.innerText = performance.responseTime.toString();
        perf.innerText = performance.mainprocessTime.toString() ?? "0";
    }, [volume, bufferingTime, performance]);

    const portrait = useMemo(() => {
        if (!selected) {
            return <></>;
        }

        const modelDir = serverSetting.serverSetting.voiceChangerParams.model_dir;
        const icon = selected.iconFile.length > 0 ? modelDir + "/" + selected.slotIndex + "/" + selected.iconFile.split(/[\/\\]/).pop() : "./assets/icons/human.png";
        const portrait = <img className="portrait" src={icon} alt={selected.name} />;
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
                        <p>
                            perf: <span id="status-perf">0</span> ms
                        </p>
                    </div>
                    <div className="portrait-area-terms-of-use">{selectedTermOfUseUrlLink}</div>
                </div>
            </div>
        );
    }, [selected]);

    return portrait;
};
