import React, { useMemo } from "react"
import { AnimationTypes, HeaderButton, HeaderButtonProps } from "../101_HeaderButton"
import { useGuiState } from "./001_GuiStateProvider"
import { InputChunkNumRow } from "./701_InputChunkNumRow"
import { ExtraDataLengthRow } from "./702_ExtraDataLengthRow"
import { GPURow } from "./703_GPURow"

export const ConverterSetting = () => {
    const guiState = useGuiState()

    const accodionButton = useMemo(() => {
        const accodionButtonProps: HeaderButtonProps = {
            stateControlCheckbox: guiState.stateControls.openConverterSettingCheckbox,
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
                {guiState.stateControls.openConverterSettingCheckbox.trigger}
                <div className="partition">
                    <div className="partition-header">
                        <span className="caret">
                            {accodionButton}
                        </span>
                        <span className="title" onClick={() => { guiState.stateControls.openConverterSettingCheckbox.updateState(!guiState.stateControls.openConverterSettingCheckbox.checked()) }}>
                            Converter Setting
                        </span>
                        <span></span>
                    </div>

                    <div className="partition-content">
                        <InputChunkNumRow />
                        <ExtraDataLengthRow />
                        <GPURow />
                    </div>
                </div>
            </>
        )
    }, [])

    return deviceSetting
}