import React, { useMemo } from "react"
import { AnimationTypes, HeaderButton, HeaderButtonProps } from "../101_HeaderButton"
import { useGuiState } from "./001_GuiStateProvider"
import { AudioInputRow } from "./401_AudioInputRow"
import { AudioInputMediaRow } from "./402_AudioInputMediaRow"
import { AudioOutputRow } from "./403_AudioOutputRow"
import { AudioOutputRecordRow } from "./404_AudioOutputRecordRow"


export const DeviceSetting = () => {
    const guiState = useGuiState()

    const accodionButton = useMemo(() => {
        const accodionButtonProps: HeaderButtonProps = {
            stateControlCheckbox: guiState.stateControls.openDeviceSettingCheckbox,
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
                {guiState.stateControls.openDeviceSettingCheckbox.trigger}
                <div className="partition">
                    <div className="partition-header">
                        <span className="caret">
                            {accodionButton}
                        </span>
                        <span className="title" onClick={() => { guiState.stateControls.openDeviceSettingCheckbox.updateState(!guiState.stateControls.openDeviceSettingCheckbox.checked()) }}>
                            Device Setting
                        </span>
                        <span></span>
                    </div>

                    <div className="partition-content">
                        <AudioInputRow />
                        <AudioInputMediaRow />
                        <AudioOutputRow />
                        <AudioOutputRecordRow />
                    </div>
                </div>
            </>
        )
    }, [])

    return deviceSetting
}