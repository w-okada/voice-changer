import React, { useMemo } from "react"
import { AnimationTypes, HeaderButton, HeaderButtonProps } from "../101_HeaderButton"
import { useGuiState } from "./001_GuiStateProvider"
import { NoiseControlRow } from "./501_NoiseControlRow"
import { GainControlRow } from "./502_GainControlRow"
import { F0DetectorRow } from "./503_F0DetectorRow"
import { AnalyzerRow } from "./510_AnalyzerRow"
import { SamplingRow } from "./511_SamplingRow"
import { SamplingPlayRow } from "./512_SamplingPlayRow"


export const QualityControl = () => {
    const guiState = useGuiState()

    const accodionButton = useMemo(() => {
        const accodionButtonProps: HeaderButtonProps = {
            stateControlCheckbox: guiState.stateControls.openQualityControlCheckbox,
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
                {guiState.stateControls.openQualityControlCheckbox.trigger}
                <div className="partition">
                    <div className="partition-header">
                        <span className="caret">
                            {accodionButton}
                        </span>
                        <span className="title" onClick={() => { guiState.stateControls.openQualityControlCheckbox.updateState(!guiState.stateControls.openQualityControlCheckbox.checked()) }}>
                            Quality Control
                        </span>
                        <span></span>
                    </div>

                    <div className="partition-content">
                        <NoiseControlRow />
                        <GainControlRow />
                        <F0DetectorRow />
                        <div className="body-row divider"></div>
                        <AnalyzerRow />
                        <SamplingRow />
                        <SamplingPlayRow />
                    </div>
                </div>
            </>
        )
    }, [])

    return deviceSetting
}