import { getLicenceInfo } from "@dannadori/voice-changer-client-js";
import React, { useMemo } from "react";
import { useAppState } from "../001_provider/001_AppStateProvider";

export const LicenseDialog = () => {
    const { frontendManagerState } = useAppState();

    const form = useMemo(() => {
        const closeButtonRow = (
            <div className="body-row split-3-4-3 left-padding-1">
                <div className="body-item-text">
                </div>
                <div className="body-button-container body-button-container-space-around">
                    <div className="body-button" onClick={() => { frontendManagerState.stateControls.showLicenseCheckbox.updateState(false) }} >close</div>
                </div>
                <div className="body-item-text"></div>
            </div>
        )
        const records = getLicenceInfo().map(x => {
            return (
                <div key={x.url} className="body-row split-3-4-3 left-padding-1">
                    <div className="body-item-text">
                        <a href={x.url} target="_blank" rel="noopener noreferrer">{x.name}</a>
                    </div>
                    <div className="body-item-text">
                        <a href={x.licenseUrl} target="_blank" rel="noopener noreferrer">{x.license}</a>
                    </div>
                    <div className="body-item-text"></div>
                </div>
            )
        })

        const records2 = (
            <>
                <div className="body-row split-3-4-3 left-padding-1">
                    <div className="body-item-text">
                        <a href="https://tyc.rei-yumesaki.net/" target="_blank" rel="noopener noreferrer">
                            つくよみちゃん
                        </a>
                    </div>
                    <div className="body-item-text">
                        コーパス
                    </div>
                    <div className="body-item-text">
                        <a href="https://tyc.rei-yumesaki.net/material/corpus/" target="_blank" rel="noopener noreferrer">
                            CV.夢前黎
                        </a>
                    </div>
                </div>
                <div className="body-row split-3-4-3 left-padding-1">
                    <div className="body-item-text">
                        <a href="https://tyc.rei-yumesaki.net/" target="_blank" rel="noopener noreferrer">
                            つくよみちゃん
                        </a>
                    </div>
                    <div className="body-item-text">
                        イラスト
                    </div>
                    <div className="body-item-text">
                        <a href="https://tyc.rei-yumesaki.net/material/illust/" target="_blank" rel="noopener noreferrer">
                            Illustration by 花兎*
                        </a>
                    </div>
                </div>
                <div className="body-row split-3-4-3 left-padding-1">
                    <div className="body-item-text">
                        <a href="https://amitaro.net/" target="_blank" rel="noopener noreferrer">
                            あみたろ
                        </a>
                    </div>
                    <div className="body-item-text">
                        音声
                    </div>
                    <div className="body-item-text">
                        <a href="https://amitaro.net/" target="_blank" rel="noopener noreferrer">
                            あみたろの声素材工房
                        </a>
                    </div>
                </div>

                <div className="body-row split-3-4-3 left-padding-1">
                    <div className="body-item-text">
                        <a href="https://amitaro.net/" target="_blank" rel="noopener noreferrer">
                            あみたろ
                        </a>
                    </div>
                    <div className="body-item-text">
                        イラスト
                    </div>
                    <div className="body-item-text">
                        <a href="" target="_blank" rel="noopener noreferrer">
                            雪透ゆき（@chira_y3）
                        </a>
                    </div>
                </div>
            </>
        )


        return (
            <div className="dialog-frame">
                <div className="dialog-title">License</div>
                <div className="dialog-content">
                    <div className={"dialog-application-title"}>Voice Changer Demo</div>
                    {records}
                    {records2}
                    {closeButtonRow}
                </div>
            </div>
        );
    }, []);
    return form;
};
