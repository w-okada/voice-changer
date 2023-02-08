import React, { useMemo } from "react";
import { AnimationTypes, HeaderButton, HeaderButtonProps } from "./101_HeaderButton";
import { useAppState } from "../../003_provider/AppStateProvider"
export const RightSidebarButton = () => {
    const { frontendManagerState } = useAppState();
    const rightSidebarButtonProps: HeaderButtonProps = {
        stateControlCheckbox: frontendManagerState.stateControls.openRightSidebarCheckbox,
        tooltip: "open/close",
        onIcon: ["fas", "angles-right"],
        offIcon: ["fas", "angles-right"],
        animation: AnimationTypes.spinner,
    };
    const rightSidebarButton = useMemo(() => {
        return <HeaderButton {...rightSidebarButtonProps}></HeaderButton>;
    }, []);
    return rightSidebarButton;
};
