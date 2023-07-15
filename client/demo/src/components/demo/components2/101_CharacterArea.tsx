import React, { useEffect, useMemo, useState } from "react";
import { useAppState } from "../../../001_provider/001_AppStateProvider";
import { useGuiState } from "../001_GuiStateProvider";
import { OnnxExporterInfo } from "@dannadori/voice-changer-client-js";
import { useMessageBuilder } from "../../../hooks/useMessageBuilder";
import { TuningArea } from "./101-1_TuningArea";
import { IndexArea } from "./101-2_IndexArea";
import { SpeakerArea } from "./101-3_SpeakerArea";
import { F0FactorArea } from "./101-4_F0FactorArea";
import { SoVitsSVC40SettingArea } from "./101-5_so-vits-svc40SettingArea";
import { DDSPSVC30SettingArea } from "./101-6_ddsp-svc30SettingArea";
import { DiffusionSVCSettingArea } from "./101-7_diffusion-svcSettingArea";

export type CharacterAreaProps = {};

export const CharacterArea = (_props: CharacterAreaProps) => {
    const { serverSetting, initializedRef, volume, bufferingTime, performance, setting, setVoiceChangerClientSetting, start, stop } = useAppState();
    const guiState = useGuiState();
    const messageBuilderState = useMessageBuilder();

    useMemo(() => {
        messageBuilderState.setMessage(__filename, "terms_of_use", { ja: "利用規約", en: "terms of use" });
        messageBuilderState.setMessage(__filename, "export_to_onnx", { ja: "onnx出力", en: "export to onnx" });
        messageBuilderState.setMessage(__filename, "save_default", { ja: "設定保存", en: "save setting" });
        messageBuilderState.setMessage(__filename, "alert_onnx", { ja: "ボイチェン中はonnx出力できません", en: "cannot export onnx when voice conversion is enabled" });
    }, []);

    const selected = useMemo(() => {
        if (serverSetting.serverSetting.modelSlotIndex == undefined) {
            return;
        }
        return serverSetting.serverSetting.modelSlots[serverSetting.serverSetting.modelSlotIndex];
    }, [serverSetting.serverSetting.modelSlotIndex, serverSetting.serverSetting.modelSlots]);

    useEffect(() => {
        const vol = document.getElementById("status-vol") as HTMLSpanElement;
        const buf = document.getElementById("status-buf") as HTMLSpanElement;
        const res = document.getElementById("status-res") as HTMLSpanElement;
        if (!vol || !buf || !res) {
            return;
        }
        vol.innerText = volume.toFixed(4);
        buf.innerText = bufferingTime.toString();
        res.innerText = performance.responseTime.toString();
    }, [volume, bufferingTime, performance]);

    const portrait = useMemo(() => {
        if (!selected) {
            return <></>;
        }

        const icon = selected.iconFile.length > 0 ? selected.iconFile : "./assets/icons/human.png";
        const selectedTermOfUseUrlLink = selected.termsOfUseUrl ? (
            <a href={selected.termsOfUseUrl} target="_blank" rel="noopener noreferrer" className="portrait-area-terms-of-use-link">
                [{messageBuilderState.getMessage(__filename, "terms_of_use")}]
            </a>
        ) : (
            <></>
        );

        return (
            <div className="portrait-area">
                <div className="portrait-container">
                    <img className="portrait" src={icon} alt={selected.name} />
                    <div className="portrait-area-status">
                        <p>
                            <span className="portrait-area-status-vctype">{selected.voiceChangerType}</span>
                        </p>
                        <p>
                            vol: <span id="status-vol">0</span>
                        </p>
                        <p>
                            buf: <span id="status-buf">0</span> ms
                        </p>
                        <p>
                            res: <span id="status-res">0</span> ms
                        </p>
                    </div>
                    <div className="portrait-area-terms-of-use">{selectedTermOfUseUrlLink}</div>
                </div>
            </div>
        );
    }, [selected]);

    const [startWithAudioContextCreate, setStartWithAudioContextCreate] = useState<boolean>(false);
    useEffect(() => {
        if (!startWithAudioContextCreate) {
            return;
        }
        guiState.setIsConverting(true);
        start();
    }, [startWithAudioContextCreate]);

    const startControl = useMemo(() => {
        const onStartClicked = async () => {
            if (serverSetting.serverSetting.enableServerAudio == 0) {
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
        const startClassName = guiState.isConverting ? "character-area-control-button-active" : "character-area-control-button-stanby";
        const stopClassName = guiState.isConverting ? "character-area-control-button-stanby" : "character-area-control-button-active";

        return (
            <div className="character-area-control">
                <div className="character-area-control-buttons">
                    <div onClick={onStartClicked} className={startClassName}>
                        start
                    </div>
                    <div onClick={onStopClicked} className={stopClassName}>
                        stop
                    </div>
                </div>
            </div>
        );
    }, [guiState.isConverting, start, stop, serverSetting.serverSetting, serverSetting.updateServerSettings]);

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
            selected.voiceChangerType == "RVC" && selected.modelFile.endsWith("pth") ? (
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
    }, [selected, serverSetting.getOnnx, serverSetting.updateModelDefault]);

    const characterArea = useMemo(() => {
        return (
            <div className="character-area">
                {portrait}
                <div className="character-area-control-area">
                    {startControl}
                    {gainControl}
                    <TuningArea />
                    <IndexArea />
                    <SpeakerArea />
                    <F0FactorArea />
                    <SoVitsSVC40SettingArea />
                    <DDSPSVC30SettingArea />
                    <DiffusionSVCSettingArea />
                    {modelSlotControl}
                </div>
            </div>
        );
    }, [portrait, startControl, gainControl, modelSlotControl]);

    return characterArea;
};
