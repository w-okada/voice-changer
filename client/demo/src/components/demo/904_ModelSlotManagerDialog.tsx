import React, { useMemo, useState } from "react";
import { useGuiState } from "./001_GuiStateProvider";
import { MainScreen } from "./904-1_MainScreen";
import { SampleDownloaderScreen } from "./904-2_SampleDownloader";
import { FileUploaderScreen } from "./904-3_FileUploader";
import { EditorScreen } from "./904-4_Editor";

export type uploadData = {
    slot: number;
    model: File | null;
    index: File | null;
};

export const ModelSlotSettingMode = {
    localFile: "localFile",
    fromNet: "fromNet",
} as const;
export type ModelSlotSettingMode = (typeof ModelSlotSettingMode)[keyof typeof ModelSlotSettingMode];

export const ModelSlotManagerDialogScreen = {
    Main: "Main",
    SampleDownloader: "SampleDownloader",
    FileUploader: "FileUploader",
    Editor: "Editor",
} as const;
export type ModelSlotManagerDialogScreen = (typeof ModelSlotManagerDialogScreen)[keyof typeof ModelSlotManagerDialogScreen];

export const ModelSlotManagerDialog = () => {
    const guiState = useGuiState();
    const [screen, setScreen] = useState<ModelSlotManagerDialogScreen>("Main");
    const [targetIndex, setTargetIndex] = useState<number>(0);

    const dialog = useMemo(() => {
        const close = () => {
            guiState.stateControls.showModelSlotManagerCheckbox.updateState(false);
        };
        const openSampleDownloader = (index: number) => {
            setTargetIndex(index);
            setScreen("SampleDownloader");
        };
        const openFileUploader = (index: number) => {
            setTargetIndex(index);
            setScreen("FileUploader");
        };
        const openEditor = (index: number) => {
            setTargetIndex(index);
            setScreen("Editor");
        };

        const backToSlotManager = () => {
            setScreen("Main");
        };
        const mainScreen = <MainScreen screen={screen} close={close} openSampleDownloader={openSampleDownloader} openFileUploader={openFileUploader} openEditor={openEditor} />;
        const sampleDownloaderScreen = <SampleDownloaderScreen screen={screen} targetIndex={targetIndex} close={close} backToSlotManager={backToSlotManager} />;
        const fileUploaderScreen = <FileUploaderScreen screen={screen} targetIndex={targetIndex} close={close} backToSlotManager={backToSlotManager} />;
        const editorScreen = <EditorScreen screen={screen} targetIndex={targetIndex} close={close} backToSlotManager={backToSlotManager} />;
        return (
            <div className="dialog-frame">
                {mainScreen}
                {sampleDownloaderScreen}
                {fileUploaderScreen}
                {editorScreen}
            </div>
        );
    }, [screen, targetIndex]);

    return dialog;
};
