import React, { useMemo } from "react"
import { AnimationTypes, HeaderButton, HeaderButtonProps } from "../101_HeaderButton"
import { useGuiState } from "./001_GuiStateProvider"
import { useAppRoot } from "../../001_provider/001_AppRootProvider"
import { generateComponent } from "./002_ComponentGenerator"


export const ModelSetting = () => {
    const guiState = useGuiState()
    const { appGuiSettingState } = useAppRoot()
    const componentSettings = appGuiSettingState.appGuiSetting.front.modelSetting
    const accodionButton = useMemo(() => {
        const accodionButtonProps: HeaderButtonProps = {
            stateControlCheckbox: guiState.stateControls.openModelSettingCheckbox,
            tooltip: "Open/Close",
            onIcon: ["fas", "caret-up"],
            offIcon: ["fas", "caret-up"],
            animation: AnimationTypes.spinner,
            tooltipClass: "tooltip-right",
        };
        return <HeaderButton {...accodionButtonProps}></HeaderButton>;
    }, []);

    const modelSetting = useMemo(() => {
        const components = componentSettings.map((x, index) => {
            const c = generateComponent(x.name, x.options)
            return <div key={`${x.name}_${index}`}>{c}</div>
        })
        return (
            <>
                {guiState.stateControls.openModelSettingCheckbox.trigger}
                <div className="partition">
                    <div className="partition-header">
                        <span className="caret">
                            {accodionButton}
                        </span>
                        <span className="title" onClick={() => { guiState.stateControls.openModelSettingCheckbox.updateState(!guiState.stateControls.openModelSettingCheckbox.checked()) }}>
                            Model Setting
                        </span>
                        <span></span>
                    </div>

                    <div className="partition-content">
                        {components}
                    </div>
                </div>
            </>
        )
    }, [])

    return modelSetting
}