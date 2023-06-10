import React from "react"
import { GuiStateProvider } from "./001_GuiStateProvider";
import { Dialogs } from "./900_Dialogs";
import { TitleArea } from "./100_TitleArea";
import { ServerControl } from "./200_ServerControl";
import { ModelSetting } from "./300_ModelSetting";
import { DeviceSetting } from "./400_DeviceSetting";
import { QualityControl } from "./500_QualityControl";
import { SpeakerSetting } from "./600_SpeakerSetting";
import { ConverterSetting } from "./700_ConverterSetting";
import { AdvancedSetting } from "./800_AdvancedSetting";
import { Lab } from "./a00_Lab";
import { ModelSlotControl } from "./b00_ModelSlotControl";
import { Dialogs2 } from "./910_Dialogs2";

export const Demo = () => {
    return (
        <GuiStateProvider>
            <div className="main-body">
                <Dialogs2 />
                <Dialogs />
                <TitleArea />
                <ModelSlotControl></ModelSlotControl>
                <ServerControl />
                <ModelSetting />
                <SpeakerSetting />
                <ConverterSetting />
                <DeviceSetting />
                <Lab />
                <QualityControl />
                <AdvancedSetting />
            </div>
        </GuiStateProvider>
    )
}