import React, { useMemo } from "react"
import { AnimationTypes, HeaderButton, HeaderButtonProps } from "../101_HeaderButton"
import { useGuiState } from "./001_GuiStateProvider"
import { StartButtonRow } from "./201_StartButtonRow"
import { PerformanceRow } from "./202_PerformanceRow"
import { ServerInfoRow } from "./203_ServerInfoRow"

export const ServerControl = () => {
    const guiState = useGuiState()

    const accodionButton = useMemo(() => {
        const accodionButtonProps: HeaderButtonProps = {
            stateControlCheckbox: guiState.stateControls.openServerControlCheckbox,
            tooltip: "Open/Close",
            onIcon: ["fas", "caret-up"],
            offIcon: ["fas", "caret-up"],
            animation: AnimationTypes.spinner,
            tooltipClass: "tooltip-right",
        };
        return <HeaderButton {...accodionButtonProps}></HeaderButton>;
    }, []);

    const serverControl = useMemo(() => {
        return (
            <>
                {guiState.stateControls.openServerControlCheckbox.trigger}
                <div className="partition">
                    <div className="partition-header">
                        <span className="caret">
                            {accodionButton}
                        </span>
                        <span className="title" onClick={() => { guiState.stateControls.openServerControlCheckbox.updateState(!guiState.stateControls.openServerControlCheckbox.checked()) }}>
                            Server Control
                        </span>
                    </div>

                    <div className="partition-content">
                        <StartButtonRow />
                        <PerformanceRow />
                        <ServerInfoRow />
                    </div>
                </div>
            </>
        )
    }, [])

    return serverControl
}