import React from "react"
import { AUDIO_ELEMENT_FOR_PLAY_RESULT, AUDIO_ELEMENT_FOR_TEST_CONVERTED, AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK, AUDIO_ELEMENT_FOR_TEST_ORIGINAL } from "../../const";
import { GuiStateProvider } from "./001_GuiStateProvider";
import { Dialogs } from "./900_Dialogs";
import { TitleArea } from "./100_TitleArea";
import { ServerControl } from "./200_ServerControl";
import { ModelSetting } from "./300_ModelSetting";
import { DeviceSetting } from "./400_DeviceSetting";
import { QualityControl } from "./500_QualityControl";



export const Demo = () => {
    return (
        <GuiStateProvider>
            <div className="main-body">
                <Dialogs />

                <TitleArea />
                <ServerControl />
                <ModelSetting />
                <DeviceSetting />
                <QualityControl />
                {/* {guiState.stateControls.openServerControlCheckbox.trigger} */}

                <audio hidden id={AUDIO_ELEMENT_FOR_PLAY_RESULT}></audio>

                org:<audio id={AUDIO_ELEMENT_FOR_TEST_ORIGINAL} controls></audio>
                <audio id={AUDIO_ELEMENT_FOR_TEST_CONVERTED} controls></audio>
                <audio id={AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK} controls hidden></audio>
            </div>
        </GuiStateProvider>

    )

}