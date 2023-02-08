import { StateControlCheckbox, useStateControlCheckbox } from "../100_components/003_hooks/useStateControlCheckbox";

export type StateControls = {
    openRightSidebarCheckbox: StateControlCheckbox
}

type FrontendManagerState = {
    stateControls: StateControls
};

export type FrontendManagerStateAndMethod = FrontendManagerState & {
}
export const useFrontendManager = (): FrontendManagerStateAndMethod => {
    // (1) Controller Switch
    const openRightSidebarCheckbox = useStateControlCheckbox("open-right-sidebar-checkbox");

    const returnValue: FrontendManagerStateAndMethod = {
        stateControls: {
            // (1) Controller Switch
            openRightSidebarCheckbox,
        },
    };
    return returnValue;
};
