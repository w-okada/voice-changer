import React, { useContext, useEffect, useState } from "react";
import { ReactNode } from "react";
import { StateControlCheckbox, useStateControlCheckbox } from "../../hooks/useStateControlCheckbox";

export const OpenServerControlCheckbox = "open-server-control-checkbox"
export const OpenModelSettingCheckbox = "open-model-setting-checkbox"
export const OpenDeviceSettingCheckbox = "open-device-setting-checkbox"
export const OpenQualityControlCheckbox = "open-quality-control-checkbox"
export const OpenSpeakerSettingCheckbox = "open-speaker-setting-checkbox"
export const OpenConverterSettingCheckbox = "open-converter-setting-checkbox"
export const OpenAdvancedSettingCheckbox = "open-advanced-setting-checkbox"

export const OpenLicenseDialogCheckbox = "open-license-dialog-checkbox"

type Props = {
    children: ReactNode;
};

export type StateControls = {
    openServerControlCheckbox: StateControlCheckbox
    openModelSettingCheckbox: StateControlCheckbox
    openDeviceSettingCheckbox: StateControlCheckbox
    openQualityControlCheckbox: StateControlCheckbox
    openSpeakerSettingCheckbox: StateControlCheckbox
    openConverterSettingCheckbox: StateControlCheckbox
    openAdvancedSettingCheckbox: StateControlCheckbox

    showLicenseCheckbox: StateControlCheckbox
}

type GuiStateAndMethod = {
    stateControls: StateControls
    isConverting: boolean,
    isAnalyzing: boolean,
    showPyTorchModelUpload: boolean
    setIsConverting: (val: boolean) => void
    setIsAnalyzing: (val: boolean) => void
    setShowPyTorchModelUpload: (val: boolean) => void

    inputAudioDeviceInfo: MediaDeviceInfo[]
    outputAudioDeviceInfo: MediaDeviceInfo[]
    audioInputForGUI: string
    audioOutputForGUI: string
    fileInputEchoback: boolean | undefined
    audioOutputForAnalyzer: string
    setInputAudioDeviceInfo: (val: MediaDeviceInfo[]) => void
    setOutputAudioDeviceInfo: (val: MediaDeviceInfo[]) => void
    setAudioInputForGUI: (val: string) => void
    setAudioOutputForGUI: (val: string) => void
    setFileInputEchoback: (val: boolean) => void
    setAudioOutputForAnalyzer: (val: string) => void
}

const GuiStateContext = React.createContext<GuiStateAndMethod | null>(null);
export const useGuiState = (): GuiStateAndMethod => {
    const state = useContext(GuiStateContext);
    if (!state) {
        throw new Error("useGuiState must be used within GuiStateProvider");
    }
    return state;
};

export const GuiStateProvider = ({ children }: Props) => {
    const [isConverting, setIsConverting] = useState<boolean>(false)
    const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false)

    const [showPyTorchModelUpload, setShowPyTorchModelUpload] = useState<boolean>(false)


    const [inputAudioDeviceInfo, setInputAudioDeviceInfo] = useState<MediaDeviceInfo[]>([])
    const [outputAudioDeviceInfo, setOutputAudioDeviceInfo] = useState<MediaDeviceInfo[]>([])
    const [audioInputForGUI, setAudioInputForGUI] = useState<string>("none")
    const [audioOutputForGUI, setAudioOutputForGUI] = useState<string>("none")
    const [fileInputEchoback, setFileInputEchoback] = useState<boolean>(false)//最初のmuteが有効になるように。undefined <-- ??? falseしておけばよさそう。undefinedだとwarningがでる。
    const [audioOutputForAnalyzer, setAudioOutputForAnalyzer] = useState<string>("default")


    const reloadDeviceInfo = async () => {
        try {
            const ms = await navigator.mediaDevices.getUserMedia({ video: false, audio: true });
            ms.getTracks().forEach(x => { x.stop() })
        } catch (e) {
            console.warn("Enumerate device error::", e)
        }
        const mediaDeviceInfos = await navigator.mediaDevices.enumerateDevices();

        const audioInputs = mediaDeviceInfos.filter(x => { return x.kind == "audioinput" })
        audioInputs.push({
            deviceId: "none",
            groupId: "none",
            kind: "audioinput",
            label: "none",
            toJSON: () => { }
        })
        audioInputs.push({
            deviceId: "file",
            groupId: "file",
            kind: "audioinput",
            label: "file",
            toJSON: () => { }
        })
        const audioOutputs = mediaDeviceInfos.filter(x => { return x.kind == "audiooutput" })
        audioOutputs.push({
            deviceId: "none",
            groupId: "none",
            kind: "audiooutput",
            label: "none",
            toJSON: () => { }
        })
        // audioOutputs.push({
        //     deviceId: "record",
        //     groupId: "record",
        //     kind: "audiooutput",
        //     label: "record",
        //     toJSON: () => { }
        // })
        return [audioInputs, audioOutputs]
    }
    useEffect(() => {
        const audioInitialize = async () => {
            const audioInfo = await reloadDeviceInfo()
            setInputAudioDeviceInfo(audioInfo[0])
            setOutputAudioDeviceInfo(audioInfo[1])
        }
        audioInitialize()
    }, [])

    // (1) Controller Switch
    const openServerControlCheckbox = useStateControlCheckbox(OpenServerControlCheckbox);
    const openModelSettingCheckbox = useStateControlCheckbox(OpenModelSettingCheckbox);
    const openDeviceSettingCheckbox = useStateControlCheckbox(OpenDeviceSettingCheckbox);
    const openQualityControlCheckbox = useStateControlCheckbox(OpenQualityControlCheckbox);
    const openSpeakerSettingCheckbox = useStateControlCheckbox(OpenSpeakerSettingCheckbox);
    const openConverterSettingCheckbox = useStateControlCheckbox(OpenConverterSettingCheckbox);
    const openAdvancedSettingCheckbox = useStateControlCheckbox(OpenAdvancedSettingCheckbox);

    const showLicenseCheckbox = useStateControlCheckbox(OpenLicenseDialogCheckbox);

    useEffect(() => {
        openServerControlCheckbox.updateState(true)
        openModelSettingCheckbox.updateState(true)
        openDeviceSettingCheckbox.updateState(true)
        openSpeakerSettingCheckbox.updateState(true)
        openConverterSettingCheckbox.updateState(true)
        openQualityControlCheckbox.updateState(true)

        showLicenseCheckbox.updateState(true)

    }, [])


    const providerValue = {
        stateControls: {
            openServerControlCheckbox,
            openModelSettingCheckbox,
            openDeviceSettingCheckbox,
            openQualityControlCheckbox,
            openSpeakerSettingCheckbox,
            openConverterSettingCheckbox,
            openAdvancedSettingCheckbox,

            showLicenseCheckbox
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
        fileInputEchoback,
        audioOutputForAnalyzer,
        setInputAudioDeviceInfo,
        setOutputAudioDeviceInfo,
        setAudioInputForGUI,
        setAudioOutputForGUI,
        setFileInputEchoback,
        setAudioOutputForAnalyzer,
    };
    return <GuiStateContext.Provider value={providerValue}>{children}</GuiStateContext.Provider>;
};



