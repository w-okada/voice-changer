
import { useState } from "react"
import { StateControlCheckbox, useStateControlCheckbox } from "../hooks/useStateControlCheckbox";
import { OpenAdvancedSettingCheckbox, OpenConverterSettingCheckbox, OpenDeviceSettingCheckbox, OpenModelSettingCheckbox, OpenQualityControlCheckbox, OpenServerControlCheckbox, OpenSpeakerSettingCheckbox } from "../const"
export type StateControls = {
    openServerControlCheckbox: StateControlCheckbox
    openModelSettingCheckbox: StateControlCheckbox
    openDeviceSettingCheckbox: StateControlCheckbox
    openQualityControlCheckbox: StateControlCheckbox
    openSpeakerSettingCheckbox: StateControlCheckbox
    openConverterSettingCheckbox: StateControlCheckbox
    openAdvancedSettingCheckbox: StateControlCheckbox
}

type FrontendManagerState = {
    stateControls: StateControls
};

export type FrontendManagerStateAndMethod = FrontendManagerState & {
}

export const useFrontendManager = (): FrontendManagerStateAndMethod => {

    // (1) Controller Switch
    const openServerControlCheckbox = useStateControlCheckbox(OpenServerControlCheckbox);
    const openModelSettingCheckbox = useStateControlCheckbox(OpenModelSettingCheckbox);
    const openDeviceSettingCheckbox = useStateControlCheckbox(OpenDeviceSettingCheckbox);
    const openQualityControlCheckbox = useStateControlCheckbox(OpenQualityControlCheckbox);
    const openSpeakerSettingCheckbox = useStateControlCheckbox(OpenSpeakerSettingCheckbox);
    const openConverterSettingCheckbox = useStateControlCheckbox(OpenConverterSettingCheckbox);
    const openAdvancedSettingCheckbox = useStateControlCheckbox(OpenAdvancedSettingCheckbox);



    const returnValue = {
        stateControls: {
            openServerControlCheckbox,
            openModelSettingCheckbox,
            openDeviceSettingCheckbox,
            openQualityControlCheckbox,
            openSpeakerSettingCheckbox,
            openConverterSettingCheckbox,
            openAdvancedSettingCheckbox
        }
    };
    return returnValue;
};
