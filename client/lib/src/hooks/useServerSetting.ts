import { useState, useMemo } from "react";
import { VoiceChangerServerSetting, ServerInfo, ServerSettingKey, OnnxExporterInfo, MergeModelRequest, VoiceChangerType, DefaultServerSetting } from "../const";
import { VoiceChangerClient } from "../VoiceChangerClient";

export const ModelAssetName = {
    iconFile: "iconFile",
} as const;
export type ModelAssetName = (typeof ModelAssetName)[keyof typeof ModelAssetName];

export const ModelFileKind = {
    mmvcv13Config: "mmvcv13Config",
    mmvcv13Model: "mmvcv13Model",
    mmvcv15Config: "mmvcv15Config",
    mmvcv15Model: "mmvcv15Model",
    mmvcv15Correspondence: "mmvcv15Correspondence",

    soVitsSvc40Config: "soVitsSvc40Config",
    soVitsSvc40Model: "soVitsSvc40Model",
    soVitsSvc40Cluster: "soVitsSvc40Cluster",

    rvcModel: "rvcModel",
    rvcIndex: "rvcIndex",

    ddspSvcModel: "ddspSvcModel",
    ddspSvcModelConfig: "ddspSvcModelConfig",
    ddspSvcDiffusion: "ddspSvcDiffusion",
    ddspSvcDiffusionConfig: "ddspSvcDiffusionConfig",

    diffusionSVCModel: "diffusionSVCModel",

    beatriceModel: "beatriceModel",

    llvcModel: "llvcModel",
    llvcConfig: "llvcConfig",
} as const;
export type ModelFileKind = (typeof ModelFileKind)[keyof typeof ModelFileKind];

export type ModelFile = {
    file: File;
    kind: ModelFileKind;
    dir: string;
};

export type ModelUploadSetting = {
    voiceChangerType: VoiceChangerType;
    slot: number;
    isSampleMode: boolean;
    sampleId: string | null;

    files: ModelFile[];
    params: any;
};
export type ModelFileForServer = Omit<ModelFile, "file"> & {
    name: string;
    kind: ModelFileKind;
};
export type ModelUploadSettingForServer = Omit<ModelUploadSetting, "files"> & {
    files: ModelFileForServer[];
};

type AssetUploadSetting = {
    slot: number;
    name: ModelAssetName;
    file: string;
};

export type UseServerSettingProps = {
    voiceChangerClient: VoiceChangerClient | null;
};

export type ServerSettingState = {
    serverSetting: ServerInfo;
    updateServerSettings: (setting: ServerInfo) => Promise<void>;
    reloadServerInfo: () => Promise<void>;

    uploadModel: (setting: ModelUploadSetting) => Promise<void>;
    uploadProgress: number;
    isUploading: boolean;

    getOnnx: () => Promise<OnnxExporterInfo>;
    mergeModel: (request: MergeModelRequest) => Promise<ServerInfo>;
    updateModelDefault: () => Promise<ServerInfo>;
    updateModelInfo: (slot: number, key: string, val: string) => Promise<ServerInfo>;
    uploadAssets: (slot: number, name: ModelAssetName, file: File) => Promise<void>;
};

export const useServerSetting = (props: UseServerSettingProps): ServerSettingState => {
    const [serverSetting, _setServerSetting] = useState<ServerInfo>(DefaultServerSetting);
    const setServerSetting = (info: ServerInfo) => {
        if (!info.modelSlots) {
            // サーバが情報を空で返したとき。Web版対策
            return;
        }
        _setServerSetting(info);
    };

    //////////////
    // 設定
    /////////////
    const updateServerSettings = useMemo(() => {
        return async (setting: ServerInfo) => {
            if (!props.voiceChangerClient) return;
            for (let i = 0; i < Object.values(ServerSettingKey).length; i++) {
                const k = Object.values(ServerSettingKey)[i] as keyof VoiceChangerServerSetting;
                const cur_v = serverSetting[k];
                const new_v = setting[k];

                if (cur_v != new_v) {
                    const res = await props.voiceChangerClient.updateServerSettings(k, "" + new_v);
                    setServerSetting(res);
                }
            }
        };
    }, [props.voiceChangerClient, serverSetting]);

    //////////////
    // 操作
    /////////////
    const [uploadProgress, setUploadProgress] = useState<number>(0);
    const [isUploading, setIsUploading] = useState<boolean>(false);

    // (e) モデルアップロード
    const _uploadFile2 = useMemo(() => {
        return async (file: File, onprogress: (progress: number, end: boolean) => void, dir: string = "") => {
            if (!props.voiceChangerClient) return;
            const num = await props.voiceChangerClient.uploadFile2(dir, file, onprogress);
            const res = await props.voiceChangerClient.concatUploadedFile(dir + file.name, num);
            console.log("uploaded", num, res);
        };
    }, [props.voiceChangerClient]);

    // 新しいアップローダ
    const uploadModel = useMemo(() => {
        return async (setting: ModelUploadSetting) => {
            if (!props.voiceChangerClient) {
                return;
            }

            setUploadProgress(0);
            setIsUploading(true);

            if (setting.isSampleMode == false) {
                const progRate = 1 / setting.files.length;
                for (let i = 0; i < setting.files.length; i++) {
                    const progOffset = 100 * i * progRate;
                    await _uploadFile2(
                        setting.files[i].file,
                        (progress: number, _end: boolean) => {
                            setUploadProgress(progress * progRate + progOffset);
                        },
                        setting.files[i].dir
                    );
                }
            }
            const params: ModelUploadSettingForServer = {
                ...setting,
                files: setting.files.map((f) => {
                    return { name: f.file.name, kind: f.kind, dir: f.dir };
                }),
            };

            const loadPromise = props.voiceChangerClient.loadModel(0, false, JSON.stringify(params));
            await loadPromise;

            setUploadProgress(0);
            setIsUploading(false);
            reloadServerInfo();
        };
    }, [props.voiceChangerClient]);

    const uploadAssets = useMemo(() => {
        return async (slot: number, name: ModelAssetName, file: File) => {
            if (!props.voiceChangerClient) return;

            await _uploadFile2(file, (progress: number, _end: boolean) => {
                console.log(progress, _end);
            });
            const assetUploadSetting: AssetUploadSetting = {
                slot,
                name,
                file: file.name,
            };
            await props.voiceChangerClient.uploadAssets(JSON.stringify(assetUploadSetting));
            reloadServerInfo();
        };
    }, [props.voiceChangerClient]);

    const reloadServerInfo = useMemo(() => {
        return async () => {
            if (!props.voiceChangerClient) return;
            const res = await props.voiceChangerClient.getServerSettings();
            setServerSetting(res);
        };
    }, [props.voiceChangerClient]);

    const getOnnx = async () => {
        return props.voiceChangerClient!.getOnnx();
    };

    const mergeModel = async (request: MergeModelRequest) => {
        const serverInfo = await props.voiceChangerClient!.mergeModel(request);
        setServerSetting(serverInfo);
        return serverInfo;
    };

    const updateModelDefault = async () => {
        const serverInfo = await props.voiceChangerClient!.updateModelDefault();
        setServerSetting(serverInfo);
        return serverInfo;
    };
    const updateModelInfo = async (slot: number, key: string, val: string) => {
        const serverInfo = await props.voiceChangerClient!.updateModelInfo(slot, key, val);
        setServerSetting(serverInfo);
        return serverInfo;
    };

    return {
        serverSetting,
        updateServerSettings,
        reloadServerInfo,

        uploadModel,
        uploadProgress,
        isUploading,
        getOnnx,
        mergeModel,
        updateModelDefault,
        updateModelInfo,
        uploadAssets,
    };
};
