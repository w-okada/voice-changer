import React, { useMemo } from "react"
import { AnimationTypes, HeaderButton, HeaderButtonProps } from "../101_HeaderButton"
import { useGuiState } from "./001_GuiStateProvider"
import { ServerURLRow } from "./801_ServerURLRow"
import { ProtocolRow } from "./802_ProtocolRow"
import { SampleRateRow } from "./803_SampleRateRow"
import { SendingSampleRateRow } from "./804_SendingSampleRateRow"
import { CrossFadeOverlapSizeRow } from "./805_CrossFadeOverlapSizeRow"
import { CrossFadeOffsetRateRow } from "./806_CrossFadeOffsetRateRow"
import { CrossFadeEndRateRow } from "./807_CrossFadeEndRateRow"
import { DownSamplingModeRow } from "./808_DownSamplingModeRow"
import { TrancateNumTresholdRow } from "./809_TrancateNumTresholdRow"

export const AdvancedSetting = () => {
    const guiState = useGuiState()

    const accodionButton = useMemo(() => {
        const accodionButtonProps: HeaderButtonProps = {
            stateControlCheckbox: guiState.stateControls.openAdvancedSettingCheckbox,
            tooltip: "Open/Close",
            onIcon: ["fas", "caret-up"],
            offIcon: ["fas", "caret-up"],
            animation: AnimationTypes.spinner,
            tooltipClass: "tooltip-right",
        };
        return <HeaderButton {...accodionButtonProps}></HeaderButton>;
    }, []);

    const deviceSetting = useMemo(() => {

        return (
            <>
                {guiState.stateControls.openAdvancedSettingCheckbox.trigger}
                <div className="partition">
                    <div className="partition-header">
                        <span className="caret">
                            {accodionButton}
                        </span>
                        <span className="title" onClick={() => { guiState.stateControls.openAdvancedSettingCheckbox.updateState(!guiState.stateControls.openAdvancedSettingCheckbox.checked()) }}>
                            Advanced Setting
                        </span>
                        <span></span>
                    </div>

                    <div className="partition-content">
                        <div className="body-row divider"></div>
                        <ServerURLRow />
                        <ProtocolRow />
                        <div className="body-row divider"></div>
                        <SampleRateRow />
                        <SendingSampleRateRow />
                        <div className="body-row divider"></div>
                        <CrossFadeOverlapSizeRow />
                        <CrossFadeOffsetRateRow />
                        <CrossFadeEndRateRow />
                        <div className="body-row divider"></div>
                        <DownSamplingModeRow />
                        <div className="body-row divider"></div>
                        <TrancateNumTresholdRow />
                    </div>
                </div>
            </>
        )
    }, [])

    return deviceSetting
}