import React, { useEffect, useMemo, useState } from "react";
import { useAppState } from "../../../001_provider/001_AppStateProvider";
import { useGuiState } from "../001_GuiStateProvider";
import { OnnxExporterInfo } from "@dannadori/voice-changer-client-js";
import { useMessageBuilder } from "../../../hooks/useMessageBuilder";
import { TuningArea } from "./101-1_TuningArea";
import { IndexArea } from "./101-2_IndexArea";
import { SpeakerArea } from "./101-3_SpeakerArea";
// import { FormantShiftArea } from "./101-4_FormantShiftArea";
import { Portrait } from "./101-0_Portrait";
import { toast } from "react-toastify";

export type CharacterAreaProps = {};

export const CharacterArea = (_props: CharacterAreaProps) => {
    const { serverSetting, initializedRef, setting, setVoiceChangerClientSetting, start, stop } = useAppState();
    const guiState = useGuiState();
    const messageBuilderState = useMessageBuilder();
    useMemo(() => {
        messageBuilderState.setMessage(__filename, "export_to_onnx", { ja: "onnx出力", en: "export to onnx" });
        messageBuilderState.setMessage(__filename, "save_default", { ja: "設定保存", en: "save setting" });
        messageBuilderState.setMessage(__filename, "alert_onnx", { ja: "ボイチェン中はonnx出力できません", en: "cannot export onnx when voice conversion is enabled" });
    }, []);

    const selected = useMemo(() => {
        if (serverSetting.serverSetting.modelSlotIndex == undefined) {
            return;
        } else {
            return serverSetting.serverSetting.modelSlots[serverSetting.serverSetting.modelSlotIndex];
        }
    }, [serverSetting.serverSetting.modelSlotIndex, serverSetting.serverSetting.modelSlots]);

    const [startWithAudioContextCreate, setStartWithAudioContextCreate] = useState<boolean>(false);
    useEffect(() => {
        if (!startWithAudioContextCreate) {
            return;
        }
        guiState.setIsConverting(true);
        start();
    }, [startWithAudioContextCreate]);

    const nameArea = useMemo(() => {
        if (!selected) {
            return <></>;
        }
        return (
            <div className="character-area-control">
                <div className="character-area-control-title">Name:</div>
                <div className="character-area-control-field">
                    <div className="character-area-text">
                        {selected.name}
                    </div>
                </div>
            </div>
        );
    }, [selected]);

    const startControl = useMemo(() => {
        const onStartClicked = async () => {
            if (serverSetting.serverSetting.modelSlotIndex === -1) {
                toast.warn('Select a voice model first.')
                return
            }
            if (serverSetting.serverSetting.enableServerAudio == 0) {
                if (!setting.voiceChangerClientSetting.audioInput || setting.voiceChangerClientSetting.audioInput == 'none') {
                    toast.warn('Select an audio input device.')
                    return
                }
                // TODO: Refactor
                if (guiState.audioOutputForGUI == 'none') {
                    toast.warn('Select an audio output device.')
                    return
                }

                if (!initializedRef.current) {
                    while (true) {
                        await new Promise<void>((resolve) => {
                            setTimeout(resolve, 500);
                        });
                        if (initializedRef.current) {
                            break;
                        }
                    }
                    setStartWithAudioContextCreate(true);
                } else {
                    guiState.setIsConverting(true);
                    await start();
                }
            } else {
                if (serverSetting.serverSetting.serverInputDeviceId == -1) {
                    toast.warn('Select an audio input device.')
                    return
                }
                if (serverSetting.serverSetting.serverOutputDeviceId == -1) {
                    toast.warn('Select an audio output device.')
                    return
                }
                serverSetting.updateServerSettings({ ...serverSetting.serverSetting, serverAudioStated: 1 });
                guiState.setIsConverting(true);
            }
        };
        const onStopClicked = async () => {
            if (serverSetting.serverSetting.enableServerAudio == 0) {
                guiState.setIsConverting(false);
                await stop();
            } else {
                guiState.setIsConverting(false);
                serverSetting.updateServerSettings({ ...serverSetting.serverSetting, serverAudioStated: 0 });
            }
        };
        const onPassThroughClicked = async () => {
            if (serverSetting.serverSetting.passThrough == false) {
                if (setting.voiceChangerClientSetting.passThroughConfirmationSkip) {
                    serverSetting.updateServerSettings({ ...serverSetting.serverSetting, passThrough: true });
                    guiState.stateControls.showEnablePassThroughDialogCheckbox.updateState(false);
                } else {
                    guiState.stateControls.showEnablePassThroughDialogCheckbox.updateState(true);
                }
            } else {
                serverSetting.updateServerSettings({ ...serverSetting.serverSetting, passThrough: false });
            }
        };
        const startClassName = guiState.isConverting ? "character-area-control-button-active" : "character-area-control-button-stanby";
        const stopClassName = guiState.isConverting ? "character-area-control-button-stanby" : "character-area-control-button-active";
        const passThruClassName = serverSetting.serverSetting.passThrough == false ? "character-area-control-passthru-button-stanby" : "character-area-control-passthru-button-active blinking";

        return (
            <div className="character-area-control">
                <div className="character-area-control-buttons">
                    <div onClick={onStartClicked} className={startClassName}>
                        start
                    </div>
                    <div onClick={onStopClicked} className={stopClassName}>
                        stop
                    </div>

                    <div onClick={onPassThroughClicked} className={passThruClassName}>
                        passthru
                    </div>
                </div>
            </div>
        );
    }, [guiState.isConverting, setting.voiceChangerClientSetting.audioInput, guiState.audioOutputForGUI, start, stop, serverSetting.serverSetting, serverSetting.updateServerSettings]);

    const gainControl = useMemo(() => {
        const currentInputGain = serverSetting.serverSetting.enableServerAudio == 0 ? setting.voiceChangerClientSetting.inputGain : serverSetting.serverSetting.serverInputAudioGain;
        const inputValueUpdatedAction =
            serverSetting.serverSetting.enableServerAudio == 0
                ? async (val: number) => {
                      await setVoiceChangerClientSetting({ ...setting.voiceChangerClientSetting, inputGain: val });
                  }
                : async (val: number) => {
                      await serverSetting.updateServerSettings({ ...serverSetting.serverSetting, serverInputAudioGain: val });
                  };

        const currentOutputGain = serverSetting.serverSetting.enableServerAudio == 0 ? setting.voiceChangerClientSetting.outputGain : serverSetting.serverSetting.serverOutputAudioGain;
        const outputValueUpdatedAction =
            serverSetting.serverSetting.enableServerAudio == 0
                ? async (val: number) => {
                      await setVoiceChangerClientSetting({ ...setting.voiceChangerClientSetting, outputGain: val });
                  }
                : async (val: number) => {
                      await serverSetting.updateServerSettings({ ...serverSetting.serverSetting, serverOutputAudioGain: val });
                  };

        return (
            <div className="character-area-control">
                <div className="character-area-control-title">GAIN:</div>
                <div className="character-area-control-field">
                    <div className="character-area-slider-control">
                        <span className="character-area-slider-control-kind">in</span>
                        <span className="character-area-slider-control-slider">
                            <input
                                type="range"
                                min="0.1"
                                max="10.0"
                                step="0.1"
                                value={currentInputGain}
                                onChange={(e) => {
                                    inputValueUpdatedAction(Number(e.target.value));
                                }}
                            ></input>
                        </span>
                        <span className="character-area-slider-control-val">{currentInputGain}</span>
                    </div>

                    <div className="character-area-slider-control">
                        <span className="character-area-slider-control-kind">out</span>
                        <span className="character-area-slider-control-slider">
                            <input
                                type="range"
                                min="0.1"
                                max="10.0"
                                step="0.1"
                                value={currentOutputGain}
                                onChange={(e) => {
                                    outputValueUpdatedAction(Number(e.target.value));
                                }}
                            ></input>
                        </span>
                        <span className="character-area-slider-control-val">{currentOutputGain}</span>
                    </div>
                </div>
            </div>
        );
    }, [serverSetting.serverSetting, setting, setVoiceChangerClientSetting, serverSetting.updateServerSettings]);

    const modelSlotControl = useMemo(() => {
        if (!selected) {
            return <></>;
        }
        const onUpdateDefaultClicked = async () => {
            await serverSetting.updateModelDefault();
        };

        const onnxExportButtonAction = async () => {
            if (guiState.isConverting) {
                alert(messageBuilderState.getMessage(__filename, "alert_onnx"));
                return;
            }

            document.getElementById("dialog")?.classList.add("dialog-container-show");
            guiState.stateControls.showWaitingCheckbox.updateState(true);
            const res = (await serverSetting.getOnnx()) as OnnxExporterInfo;
            const a = document.createElement("a");
            a.href = res.path;
            a.download = res.filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            guiState.stateControls.showWaitingCheckbox.updateState(false);
        };

        const exportOnnx =
            selected.voiceChangerType == "RVC" && (selected.modelFile.endsWith("pth") || selected.modelFile.endsWith("safetensors")) ? (
                <div className="character-area-button" onClick={onnxExportButtonAction}>
                    {messageBuilderState.getMessage(__filename, "export_to_onnx")}
                </div>
            ) : (
                <></>
            );
        return (
            <div className="character-area-control">
                <div className="character-area-control-title"></div>
                <div className="character-area-control-field">
                    <div className="character-area-buttons">
                        <div className="character-area-button" onClick={onUpdateDefaultClicked}>
                            {messageBuilderState.getMessage(__filename, "save_default")}
                        </div>
                        {exportOnnx}
                    </div>
                </div>
            </div>
        );
    }, [selected, serverSetting.getOnnx, serverSetting.updateModelDefault, guiState.isConverting]);

    const characterArea = useMemo(() => {
        return (
            <div className="character-area">
                <Portrait></Portrait>
                <div className="character-area-control-area">
                    {nameArea}
                    {startControl}
                    {gainControl}
                    <TuningArea />
                    <IndexArea />
                    <SpeakerArea />
                    {modelSlotControl}
                </div>
            </div>
        );
    }, [startControl, gainControl, modelSlotControl]);

    return characterArea;
};
