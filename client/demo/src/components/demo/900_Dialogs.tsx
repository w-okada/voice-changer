import React from "react";
import { useGuiState } from "./001_GuiStateProvider";
import { WaitingDialog } from "./902_WaitingDialog";
import { StartingNoticeDialog } from "./903_StartingNoticeDialog";
import { ModelSlotManagerDialog } from "./904_ModelSlotManagerDialog";
import { MergeLabDialog } from "./905_MergeLabDialog";
import { AdvancedSettingDialog } from "./906_AdvancedSettingDialog";
import { GetServerInfomationDialog } from "./907_GetServerInfomationDialog";
import { GetClientInfomationDialog } from "./908_GetClientInfomationDialog";

export const Dialogs = () => {
    const guiState = useGuiState();
    const dialogs = (
        <div>
            {guiState.stateControls.showWaitingCheckbox.trigger}
            {guiState.stateControls.showStartingNoticeCheckbox.trigger}
            {guiState.stateControls.showModelSlotManagerCheckbox.trigger}
            {guiState.stateControls.showMergeLabCheckbox.trigger}
            {guiState.stateControls.showAdvancedSettingCheckbox.trigger}
            {guiState.stateControls.showGetServerInformationCheckbox.trigger}
            {guiState.stateControls.showGetClientInformationCheckbox.trigger}
            <div className="dialog-container" id="dialog">
                {guiState.stateControls.showWaitingCheckbox.trigger}
                <WaitingDialog></WaitingDialog>
                {guiState.stateControls.showStartingNoticeCheckbox.trigger}
                <StartingNoticeDialog></StartingNoticeDialog>
                {guiState.stateControls.showModelSlotManagerCheckbox.trigger}
                <ModelSlotManagerDialog></ModelSlotManagerDialog>
                {guiState.stateControls.showMergeLabCheckbox.trigger}
                <MergeLabDialog></MergeLabDialog>
                {guiState.stateControls.showAdvancedSettingCheckbox.trigger}
                <AdvancedSettingDialog></AdvancedSettingDialog>
                {guiState.stateControls.showGetServerInformationCheckbox.trigger}
                <GetServerInfomationDialog></GetServerInfomationDialog>
                {guiState.stateControls.showGetClientInformationCheckbox.trigger}
                <GetClientInfomationDialog></GetClientInfomationDialog>
            </div>
        </div>
    );

    return dialogs;
};
