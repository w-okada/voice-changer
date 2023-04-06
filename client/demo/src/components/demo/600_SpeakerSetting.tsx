import React, { useMemo } from "react"
import { AnimationTypes, HeaderButton, HeaderButtonProps } from "../101_HeaderButton"
import { useGuiState } from "./001_GuiStateProvider"
import { SrcIdRow } from "./601_SrcIdRow"
import { DstIdRow } from "./602_DstIdRow"
import { EditSpeakerIdMappingRow } from "./603_EditSpeakerIdMappingRow"
import { F0FactorRow } from "./604_F0FactorRow"
import { TuneRow } from "./605_TuneRow"
import { ClusterInferRatioRow } from "./606_ClusterInferRatioRow"
import { NoiseScaleRow } from "./607_NoiseScaleRow"
import { SilentThresholdRow } from "./608_SilentThresholdRow"


export const SpeakerSetting = () => {
    const guiState = useGuiState()

    const accodionButton = useMemo(() => {
        const accodionButtonProps: HeaderButtonProps = {
            stateControlCheckbox: guiState.stateControls.openSpeakerSettingCheckbox,
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
                {guiState.stateControls.openSpeakerSettingCheckbox.trigger}
                <div className="partition">
                    <div className="partition-header">
                        <span className="caret">
                            {accodionButton}
                        </span>
                        <span className="title" onClick={() => { guiState.stateControls.openSpeakerSettingCheckbox.updateState(!guiState.stateControls.openSpeakerSettingCheckbox.checked()) }}>
                            Speaker Setting
                        </span>
                        <span></span>
                    </div>

                    <div className="partition-content">
                        <SrcIdRow />
                        <DstIdRow />
                        <EditSpeakerIdMappingRow />
                        <F0FactorRow />
                        <TuneRow />
                        <ClusterInferRatioRow />
                        <NoiseScaleRow />
                        <SilentThresholdRow />
                    </div>
                </div>
            </>
        )
    }, [])

    return deviceSetting
}