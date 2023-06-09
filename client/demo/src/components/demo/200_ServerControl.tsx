import React, { useMemo } from "react"
import { useAppRoot } from "../../001_provider/001_AppRootProvider"
import { AnimationTypes, HeaderButton, HeaderButtonProps } from "../101_HeaderButton"
import { useGuiState } from "./001_GuiStateProvider"
import { generateComponent } from "./002_ComponentGenerator"

export const ServerControl = () => {
    const guiState = useGuiState()
    const { appGuiSettingState } = useAppRoot()
    const componentSettings = appGuiSettingState.appGuiSetting.front.serverControl

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
        if (componentSettings.length == 0) {
            return <></>
        }
        const components = componentSettings.map((x, index) => {
            const c = generateComponent(x.name, x.options)
            return <div key={`${x.name}_${index}`}>{c}</div>
        })
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
                        {components}
                    </div>
                </div>
            </>
        )
    }, [])

    return serverControl
}