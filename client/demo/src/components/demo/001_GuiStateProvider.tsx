import React, { useContext, useEffect, useState, useRef } from "react";
import { ReactNode } from "react";
import { useAppRoot } from "../../001_provider/001_AppRootProvider";
import { StateControlCheckbox, useStateControlCheckbox } from "../../hooks/useStateControlCheckbox";
import { useAppState } from "../../001_provider/001_AppStateProvider";

export const OpenServerControlCheckbox = "open-server-control-checkbox";
export const OpenModelSettingCheckbox = "open-model-setting-checkbox";
export const OpenDeviceSettingCheckbox = "open-device-setting-checkbox";
export const OpenQualityControlCheckbox = "open-quality-control-checkbox";
export const OpenSpeakerSettingCheckbox = "open-speaker-setting-checkbox";
export const OpenConverterSettingCheckbox = "open-converter-setting-checkbox";
export const OpenAdvancedSettingCheckbox = "open-advanced-setting-checkbox";
export const OpenLabCheckbox = "open-lab-checkbox";

export const OpenLicenseDialogCheckbox = "open-license-dialog-checkbox";
export const OpenWaitingDialogCheckbox = "open-waiting-dialog-checkbox";
export const OpenStartingNoticeDialogCheckbox = "open-starting-notice-dialog-checkbox";
export const OpenModelSlotManagerDialogCheckbox = "open-model-slot-manager-dialog-checkbox";
export const OpenMergeLabDialogCheckbox = "open-merge-lab-dialog-checkbox";
export const OpenAdvancedSettingDialogCheckbox = "open-advanced-setting-dialog-checkbox";
export const OpenGetServerInformationDialogCheckbox = "open-get-server-information-dialog-checkbox";
export const OpenGetClientInformationDialogCheckbox = "open-get-client-information-dialog-checkbox";
export const OpenEnablePassThroughDialogCheckbox = "open-enable-pass-through-dialog-checkbox";

export const OpenTextInputDialogCheckbox = "open-text-input-dialog-checkbox";
export const OpenShowLicenseDialogCheckbox = "open-show-license-dialog-checkbox";

type Props = {
    children: ReactNode;
};

export type StateControls = {
    openServerControlCheckbox: StateControlCheckbox;
    openModelSettingCheckbox: StateControlCheckbox;
    openDeviceSettingCheckbox: StateControlCheckbox;
    openQualityControlCheckbox: StateControlCheckbox;
    openSpeakerSettingCheckbox: StateControlCheckbox;
    openConverterSettingCheckbox: StateControlCheckbox;
    openAdvancedSettingCheckbox: StateControlCheckbox;
    openLabCheckbox: StateControlCheckbox;

    showWaitingCheckbox: StateControlCheckbox;
    showStartingNoticeCheckbox: StateControlCheckbox;
    showModelSlotManagerCheckbox: StateControlCheckbox;

    showMergeLabCheckbox: StateControlCheckbox;
    showAdvancedSettingCheckbox: StateControlCheckbox;
    showGetServerInformationCheckbox: StateControlCheckbox;
    showGetClientInformationCheckbox: StateControlCheckbox;
    showEnablePassThroughDialogCheckbox: StateControlCheckbox;
    showTextInputCheckbox: StateControlCheckbox;
    showLicenseCheckbox: StateControlCheckbox;
};

type GuiStateAndMethod = {
    stateControls: StateControls;
    isConverting: boolean;
    isAnalyzing: boolean;
    showPyTorchModelUpload: boolean;
    setIsConverting: (val: boolean) => void;
    setIsAnalyzing: (val: boolean) => void;
    setShowPyTorchModelUpload: (val: boolean) => void;

    reloadDeviceInfo: () => Promise<void>;
    inputAudioDeviceInfo: MediaDeviceInfo[];
    outputAudioDeviceInfo: MediaDeviceInfo[];
    audioInputForGUI: string;
    audioOutputForGUI: string;
    audioMonitorForGUI: string;
    fileInputEchoback: boolean | undefined;
    shareScreenEnabled: boolean;
    audioOutputForAnalyzer: string;
    setInputAudioDeviceInfo: (val: MediaDeviceInfo[]) => void;
    setOutputAudioDeviceInfo: (val: MediaDeviceInfo[]) => void;
    setAudioInputForGUI: (val: string) => void;
    setAudioOutputForGUI: (val: string) => void;
    setAudioMonitorForGUI: (val: string) => void;
    setFileInputEchoback: (val: boolean) => void;
    setShareScreenEnabled: (val: boolean) => void;
    setAudioOutputForAnalyzer: (val: string) => void;

    modelSlotNum: number;
    setModelSlotNum: (val: number) => void;

    textInputResolve: TextInputResolveType | null;
    setTextInputResolve: (val: TextInputResolveType | null) => void;
};

const GuiStateContext = React.createContext<GuiStateAndMethod | null>(null);
export const useGuiState = (): GuiStateAndMethod => {
    const state = useContext(GuiStateContext);
    if (!state) {
        throw new Error("useGuiState must be used within GuiStateProvider");
    }
    return state;
};

type TextInputResolveType = {
    resolve: ((value: string | PromiseLike<string>) => void) | null;
};

export const GuiStateProvider = ({ children }: Props) => {
    const { appGuiSettingState } = useAppRoot();
    const { serverSetting } = useAppState();
    const [isConverting, setIsConverting] = useState<boolean>(false);
    const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
    const [modelSlotNum, setModelSlotNum] = useState<number>(0);

    const [showPyTorchModelUpload, setShowPyTorchModelUpload] = useState<boolean>(false);

    const [inputAudioDeviceInfo, setInputAudioDeviceInfo] = useState<MediaDeviceInfo[]>([]);
    const [outputAudioDeviceInfo, setOutputAudioDeviceInfo] = useState<MediaDeviceInfo[]>([]);
    const [audioInputForGUI, setAudioInputForGUI] = useState<string>("none");
    const [audioOutputForGUI, setAudioOutputForGUI] = useState<string>("none");
    const [audioMonitorForGUI, setAudioMonitorForGUI] = useState<string>("none");
    const [fileInputEchoback, setFileInputEchoback] = useState<boolean>(false); //最初のmuteが有効になるように。undefined <-- ??? falseしておけばよさそう。undefinedだとwarningがでる。
    const [shareScreenEnabled, setShareScreenEnabled] = useState<boolean>(false);
    const [audioOutputForAnalyzer, setAudioOutputForAnalyzer] = useState<string>("default");

    const [textInputResolve, setTextInputResolve] = useState<TextInputResolveType | null>(null);

    const checkDeviceAvailable = useRef<boolean>(false);

    const _reloadDeviceInfo = async () => {
        // デバイスチェックの空振り
        if (checkDeviceAvailable.current == false) {
            try {
                const ms = await navigator.mediaDevices.getUserMedia({ video: false, audio: true });
                ms.getTracks().forEach((x) => {
                    x.stop();
                });
                checkDeviceAvailable.current = true;
            } catch (e) {
                console.warn("Enumerate device error::", e);
            }
        }
        const mediaDeviceInfos = await navigator.mediaDevices.enumerateDevices();

        const audioInputs = mediaDeviceInfos.filter((x) => {
            return x.kind == "audioinput";
        });
        audioInputs.push({
            deviceId: "none",
            groupId: "none",
            kind: "audioinput",
            label: "none",
            toJSON: () => {},
        });
        audioInputs.push({
            deviceId: "file",
            groupId: "file",
            kind: "audioinput",
            label: "file",
            toJSON: () => {},
        });
        audioInputs.push({
            deviceId: "screen",
            groupId: "screen",
            kind: "audioinput",
            label: "system(only win)",
            toJSON: () => {},
        });
        const audioOutputs = mediaDeviceInfos.filter((x) => {
            return x.kind == "audiooutput";
        });
        audioOutputs.push({
            deviceId: "none",
            groupId: "none",
            kind: "audiooutput",
            label: "none",
            toJSON: () => {},
        });
        // audioOutputs.push({
        //     deviceId: "record",
        //     groupId: "record",
        //     kind: "audiooutput",
        //     label: "record",
        //     toJSON: () => { }
        // })
        return [audioInputs, audioOutputs];
    };

    const reloadDeviceInfo = async () => {
        const audioInfo = await _reloadDeviceInfo();
        setInputAudioDeviceInfo(audioInfo[0]);
        setOutputAudioDeviceInfo(audioInfo[1]);
    };

    // useEffect(() => {
    //     const audioInitialize = async () => {
    //         await reloadDeviceInfo();
    //     };
    //     audioInitialize();
    // }, []);

    useEffect(() => {
        let isMounted = true;

        // デバイスのポーリングを再帰的に実行する関数
        const pollDevices = async () => {
            const checkDeviceDiff = (knownDeviceIds: Set<string>, newDeviceIds: Set<string>) => {
                const deleted = new Set([...knownDeviceIds].filter((x) => !newDeviceIds.has(x)));
                const added = new Set([...newDeviceIds].filter((x) => !knownDeviceIds.has(x)));
                return { deleted, added };
            };
            try {
                const audioInfo = await _reloadDeviceInfo();

                const knownAudioinputIds = new Set(inputAudioDeviceInfo.map((x) => x.deviceId));
                const newAudioinputIds = new Set(audioInfo[0].map((x) => x.deviceId));

                const knownAudiooutputIds = new Set(outputAudioDeviceInfo.map((x) => x.deviceId));
                const newAudiooutputIds = new Set(audioInfo[1].map((x) => x.deviceId));

                const audioInputDiff = checkDeviceDiff(knownAudioinputIds, newAudioinputIds);
                const audioOutputDiff = checkDeviceDiff(knownAudiooutputIds, newAudiooutputIds);

                if (audioInputDiff.deleted.size > 0 || audioInputDiff.added.size > 0) {
                    console.log(`deleted input device: ${[...audioInputDiff.deleted]}`);
                    console.log(`added input device: ${[...audioInputDiff.added]}`);
                    setInputAudioDeviceInfo(audioInfo[0]);
                }
                if (audioOutputDiff.deleted.size > 0 || audioOutputDiff.added.size > 0) {
                    console.log(`deleted output device: ${[...audioOutputDiff.deleted]}`);
                    console.log(`added output device: ${[...audioOutputDiff.added]}`);
                    setOutputAudioDeviceInfo(audioInfo[1]);
                }

                if (isMounted) {
                    setTimeout(pollDevices, 1000 * 3);
                }
            } catch (err) {
                console.error("An error occurred during enumeration of devices:", err);
            }
        };

        pollDevices();
        return () => {
            isMounted = false;
        };
    }, [inputAudioDeviceInfo, outputAudioDeviceInfo]);

    // (1) Controller Switch
    const openServerControlCheckbox = useStateControlCheckbox(OpenServerControlCheckbox);
    const openModelSettingCheckbox = useStateControlCheckbox(OpenModelSettingCheckbox);
    const openDeviceSettingCheckbox = useStateControlCheckbox(OpenDeviceSettingCheckbox);
    const openQualityControlCheckbox = useStateControlCheckbox(OpenQualityControlCheckbox);
    const openSpeakerSettingCheckbox = useStateControlCheckbox(OpenSpeakerSettingCheckbox);
    const openConverterSettingCheckbox = useStateControlCheckbox(OpenConverterSettingCheckbox);
    const openAdvancedSettingCheckbox = useStateControlCheckbox(OpenAdvancedSettingCheckbox);
    const openLabCheckbox = useStateControlCheckbox(OpenLabCheckbox);

    const showWaitingCheckbox = useStateControlCheckbox(OpenWaitingDialogCheckbox);
    const showStartingNoticeCheckbox = useStateControlCheckbox(OpenStartingNoticeDialogCheckbox);
    const showModelSlotManagerCheckbox = useStateControlCheckbox(OpenModelSlotManagerDialogCheckbox);
    const showMergeLabCheckbox = useStateControlCheckbox(OpenMergeLabDialogCheckbox);
    const showAdvancedSettingCheckbox = useStateControlCheckbox(OpenAdvancedSettingDialogCheckbox);
    const showGetServerInformationCheckbox = useStateControlCheckbox(OpenGetServerInformationDialogCheckbox);
    const showGetClientInformationCheckbox = useStateControlCheckbox(OpenGetClientInformationDialogCheckbox);
    const showEnablePassThroughDialogCheckbox = useStateControlCheckbox(OpenEnablePassThroughDialogCheckbox);

    const showTextInputCheckbox = useStateControlCheckbox(OpenTextInputDialogCheckbox);
    const showLicenseCheckbox = useStateControlCheckbox(OpenShowLicenseDialogCheckbox);

    useEffect(() => {
        openServerControlCheckbox.updateState(true);
        openModelSettingCheckbox.updateState(false);
        openDeviceSettingCheckbox.updateState(true);
        openSpeakerSettingCheckbox.updateState(true);
        openConverterSettingCheckbox.updateState(true);
        openQualityControlCheckbox.updateState(false);
        openLabCheckbox.updateState(false);
        openAdvancedSettingCheckbox.updateState(false);

        showWaitingCheckbox.updateState(false);

        showStartingNoticeCheckbox.updateState(false);
        showModelSlotManagerCheckbox.updateState(false);
        showMergeLabCheckbox.updateState(false);
        showAdvancedSettingCheckbox.updateState(false);
        showGetServerInformationCheckbox.updateState(false);
        showGetClientInformationCheckbox.updateState(false);
        showEnablePassThroughDialogCheckbox.updateState(false);

        showTextInputCheckbox.updateState(false);
        showLicenseCheckbox.updateState(false);
    }, []);

    useEffect(() => {
        const show = () => {
            // const lang = window.navigator.language
            // const edition = appGuiSettingState.edition
            // console.log("appGuiSettingState.edition", appGuiSettingState.edition, lang)
            // if ((edition == "onnxdirectML-cuda" || edition == "") && lang == "ja") {
            //     return
            // }

            document.getElementById("dialog")?.classList.add("dialog-container-show");
            showStartingNoticeCheckbox.updateState(true);
            document.getElementById("dialog2")?.classList.add("dialog-container-show");
        };
        setTimeout(show);
    }, [appGuiSettingState.edition]);

    const providerValue: GuiStateAndMethod = {
        stateControls: {
            openServerControlCheckbox,
            openModelSettingCheckbox,
            openDeviceSettingCheckbox,
            openQualityControlCheckbox,
            openSpeakerSettingCheckbox,
            openConverterSettingCheckbox,
            openAdvancedSettingCheckbox,
            openLabCheckbox,

            showWaitingCheckbox,
            showStartingNoticeCheckbox,
            showModelSlotManagerCheckbox,

            showMergeLabCheckbox,
            showAdvancedSettingCheckbox,
            showGetServerInformationCheckbox,
            showGetClientInformationCheckbox,
            showEnablePassThroughDialogCheckbox,

            showTextInputCheckbox,
            showLicenseCheckbox,
        },
        isConverting,
        setIsConverting,
        isAnalyzing,
        setIsAnalyzing,
        showPyTorchModelUpload,
        setShowPyTorchModelUpload,

        reloadDeviceInfo,
        inputAudioDeviceInfo,
        outputAudioDeviceInfo,
        audioInputForGUI,
        audioOutputForGUI,
        audioMonitorForGUI,
        fileInputEchoback,
        shareScreenEnabled,
        audioOutputForAnalyzer,
        setInputAudioDeviceInfo,
        setOutputAudioDeviceInfo,
        setAudioInputForGUI,
        setAudioOutputForGUI,
        setAudioMonitorForGUI,
        setFileInputEchoback,
        setShareScreenEnabled,
        setAudioOutputForAnalyzer,

        modelSlotNum,
        setModelSlotNum,

        textInputResolve,
        setTextInputResolve,
    };
    return <GuiStateContext.Provider value={providerValue}>{children}</GuiStateContext.Provider>;
};
