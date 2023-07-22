import React, { useEffect, useMemo, useState } from "react";
import { useAppState } from "../../001_provider/001_AppStateProvider";
import { ModelFileKind, ModelUploadSetting, VoiceChangerType, fileSelector } from "@dannadori/voice-changer-client-js";
import { useMessageBuilder } from "../../hooks/useMessageBuilder";
import { ModelSlotManagerDialogScreen } from "./904_ModelSlotManagerDialog";
import { checkExtention, trimfileName } from "../../utils/utils";

export type FileUploaderScreenProps = {
    screen: ModelSlotManagerDialogScreen;
    targetIndex: number;
    close: () => void;
    backToSlotManager: () => void;
};

export const FileUploaderScreen = (props: FileUploaderScreenProps) => {
    const { serverSetting } = useAppState();
    const [voiceChangerType, setVoiceChangerType] = useState<VoiceChangerType>("RVC");
    const [uploadSetting, setUploadSetting] = useState<ModelUploadSetting>();
    const messageBuilderState = useMessageBuilder();

    useMemo(() => {
        messageBuilderState.setMessage(__filename, "header_message", { ja: "ファイルをアップロードしてください. 対象：", en: "Upload Files for " });
        messageBuilderState.setMessage(__filename, "back", { ja: "戻る", en: "back" });
        messageBuilderState.setMessage(__filename, "select", { ja: "ファイル選択", en: "select file" });
        messageBuilderState.setMessage(__filename, "upload", { ja: "アップロード", en: "upload" });
        messageBuilderState.setMessage(__filename, "uploading", { ja: "アップロード中", en: "uploading" });
        messageBuilderState.setMessage(__filename, "alert-model-ext", {
            ja: "ファイルの拡張子は次のモノである必要があります。",
            en: "extension of file should be the following.",
        });
        messageBuilderState.setMessage(__filename, "alert-model-file", {
            ja: "ファイルが選択されていません",
            en: "file is not selected.",
        });
    }, []);

    useEffect(() => {
        setUploadSetting({
            voiceChangerType: voiceChangerType,
            slot: props.targetIndex,
            isSampleMode: false,
            sampleId: null,
            files: [],
            params: {},
        });
    }, [props.targetIndex, voiceChangerType]);

    const screen = useMemo(() => {
        if (props.screen != "FileUploader") {
            return <></>;
        }

        const vcTypeOptions = Object.values(VoiceChangerType).map((x) => {
            return (
                <option key={x} value={x}>
                    {x}
                </option>
            );
        });

        const checkModelSetting = (setting: ModelUploadSetting) => {
            if (setting.voiceChangerType == "RVC") {
                const enough = !!setting.files.find((x) => {
                    return x.kind == "rvcModel";
                });
                return enough;
            } else if (setting.voiceChangerType == "MMVCv13") {
                const enough =
                    !!setting.files.find((x) => {
                        return x.kind == "mmvcv13Model";
                    }) &&
                    !!setting.files.find((x) => {
                        return x.kind == "mmvcv13Config";
                    });
                return enough;
            } else if (setting.voiceChangerType == "MMVCv15") {
                const enough =
                    !!setting.files.find((x) => {
                        return x.kind == "mmvcv15Model";
                    }) &&
                    !!setting.files.find((x) => {
                        return x.kind == "mmvcv15Config";
                    });
                return enough;
            } else if (setting.voiceChangerType == "so-vits-svc-40") {
                const enough =
                    !!setting.files.find((x) => {
                        return x.kind == "soVitsSvc40Config";
                    }) &&
                    !!setting.files.find((x) => {
                        return x.kind == "soVitsSvc40Model";
                    });
                return enough;
            } else if (setting.voiceChangerType == "DDSP-SVC") {
                const enough =
                    !!setting.files.find((x) => {
                        return x.kind == "ddspSvcModel";
                    }) &&
                    !!setting.files.find((x) => {
                        return x.kind == "ddspSvcModelConfig";
                    }) &&
                    !!setting.files.find((x) => {
                        return x.kind == "ddspSvcDiffusion";
                    }) &&
                    !!setting.files.find((x) => {
                        return x.kind == "ddspSvcDiffusionConfig";
                    });
                return enough;
            } else if (setting.voiceChangerType == "Diffusion-SVC") {
                const enough = !!setting.files.find((x) => {
                    return x.kind == "diffusionSVCModel";
                });
                return enough;
            }
            return false;
        };

        const generateFileRow = (setting: ModelUploadSetting, title: string, kind: ModelFileKind, ext: string[], dir: string = "") => {
            const selectedFile = setting.files.find((x) => {
                return x.kind == kind;
            });
            const selectedFilename = selectedFile?.file.name || "";
            return (
                <div key={`${title}`} className="file-uploader-file-select-row">
                    <div className="file-uploader-file-select-row-label">{title}:</div>
                    <div className="file-uploader-file-select-row-value">{trimfileName(selectedFilename, 30)}</div>
                    <div
                        className="file-uploader-file-select-row-button"
                        onClick={async () => {
                            const file = await fileSelector("");
                            if (checkExtention(file.name, ext) == false) {
                                const alertMessage = `${messageBuilderState.getMessage(__filename, "alert-model-ext")} ${ext}`;
                                alert(alertMessage);
                                return;
                            }
                            if (selectedFile) {
                                selectedFile.file = file;
                            } else {
                                setting.files.push({ kind: kind, file: file, dir: dir });
                            }
                            setUploadSetting({ ...setting });
                        }}
                    >
                        {messageBuilderState.getMessage(__filename, "select")}
                    </div>
                </div>
            );
        };

        const generateFileRowsByVCType = (vcType: VoiceChangerType) => {
            const rows: JSX.Element[] = [];
            if (vcType == "RVC") {
                rows.push(generateFileRow(uploadSetting!, "Model", "rvcModel", ["pth", "onnx"]));
                rows.push(generateFileRow(uploadSetting!, "Index", "rvcIndex", ["index", "bin"]));
            } else if (vcType == "MMVCv13") {
                rows.push(generateFileRow(uploadSetting!, "Config", "mmvcv13Config", ["json"]));
                rows.push(generateFileRow(uploadSetting!, "Model", "mmvcv13Model", ["pth", "onnx"]));
            } else if (vcType == "MMVCv15") {
                rows.push(generateFileRow(uploadSetting!, "Config", "mmvcv15Config", ["json"]));
                rows.push(generateFileRow(uploadSetting!, "Corres", "mmvcv15Correspondence", ["txt"]));
                rows.push(generateFileRow(uploadSetting!, "Model", "mmvcv15Model", ["pth", "onnx"]));
            } else if (vcType == "so-vits-svc-40") {
                rows.push(generateFileRow(uploadSetting!, "Config", "soVitsSvc40Config", ["json"]));
                rows.push(generateFileRow(uploadSetting!, "Model", "soVitsSvc40Model", ["pth"]));
                rows.push(generateFileRow(uploadSetting!, "Cluster", "soVitsSvc40Cluster", ["pth", "pt"]));
            } else if (vcType == "DDSP-SVC") {
                rows.push(generateFileRow(uploadSetting!, "Config", "ddspSvcModelConfig", ["yaml"], "model/"));
                rows.push(generateFileRow(uploadSetting!, "Model", "ddspSvcModel", ["pth", "pt"], "model/"));
                rows.push(generateFileRow(uploadSetting!, "Config(diff)", "ddspSvcDiffusionConfig", ["yaml"], "diff/"));
                rows.push(generateFileRow(uploadSetting!, "Model(diff)", "ddspSvcDiffusion", ["pth", "pt"], "diff/"));
            } else if (vcType == "Diffusion-SVC") {
                rows.push(generateFileRow(uploadSetting!, "Model(combo)", "diffusionSVCModel", ["ptc"]));
            }
            return rows;
        };
        const fileRows = generateFileRowsByVCType(voiceChangerType);

        // appState.serverSetting.uploadProgress == 0 ? `loading model...(wait about 20sec)` : `processing.... ${appState.serverSetting.uploadProgress.toFixed(1)}%` : ""

        const buttonLabel = serverSetting.uploadProgress == 0 ? messageBuilderState.getMessage(__filename, "upload") : messageBuilderState.getMessage(__filename, "uploading") + `(${serverSetting.uploadProgress.toFixed(1)}%)`;
        return (
            <div className="dialog-frame">
                <div className="dialog-title">File Uploader</div>
                <div className="dialog-fixed-size-content">
                    <div className="file-uploader-header">
                        {messageBuilderState.getMessage(__filename, "header_message")} Slot[{props.targetIndex}]
                        <span
                            onClick={() => {
                                props.backToSlotManager();
                            }}
                            className="file-uploader-header-button"
                        >
                            &lt;&lt;{messageBuilderState.getMessage(__filename, "back")}
                        </span>
                    </div>
                    <div className="file-uploader-voice-changer-select">
                        VoiceChangerType:
                        <select
                            value={voiceChangerType}
                            onChange={(e) => {
                                setVoiceChangerType(e.target.value as VoiceChangerType);
                            }}
                        >
                            {vcTypeOptions}
                        </select>
                    </div>

                    <div className="file-uploader-file-select-container">{fileRows}</div>
                    <div className="file-uploader-file-select-upload-button-container">
                        <div
                            className="file-uploader-file-select-upload-button"
                            onClick={() => {
                                if (!uploadSetting) {
                                    return;
                                }
                                if (serverSetting.uploadProgress != 0) {
                                    return;
                                }
                                if (checkModelSetting(uploadSetting)) {
                                    serverSetting.uploadModel(uploadSetting).then(() => {
                                        props.backToSlotManager();
                                    });
                                } else {
                                    const errorMessage = messageBuilderState.getMessage(__filename, "alert-model-file");
                                    alert(errorMessage);
                                }
                            }}
                        >
                            {buttonLabel}
                        </div>
                    </div>
                </div>
            </div>
        );
    }, [props.screen, props.targetIndex, voiceChangerType, uploadSetting, serverSetting.uploadModel, serverSetting.uploadProgress]);

    return screen;
};
