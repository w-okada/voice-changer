import { useState, useMemo, useEffect } from "react"
import { VoiceChangerServerSetting, ServerInfo, ServerSettingKey, INDEXEDDB_KEY_SERVER, INDEXEDDB_KEY_MODEL_DATA, ClientType, DefaultServerSetting_MMVCv13, DefaultServerSetting_MMVCv15, DefaultServerSetting_so_vits_svc_40v2, DefaultServerSetting_so_vits_svc_40, DefaultServerSetting_so_vits_svc_40_c, DefaultServerSetting_RVC, OnnxExporterInfo, DefaultServerSetting_DDSP_SVC, MAX_MODEL_SLOT_NUM, Framework, MergeModelRequest, VoiceChangerType } from "../const"
import { VoiceChangerClient } from "../VoiceChangerClient"
import { useIndexedDB } from "./useIndexedDB"
import { ModelLoadException } from "../exceptions"


type ModelData = {
    file?: File
    data?: ArrayBuffer
    filename?: string
}

export const ModelAssetName = {
    iconFile: "iconFile"
} as const
export type ModelAssetName = typeof ModelAssetName[keyof typeof ModelAssetName]


export const ModelFileKind = {
    "mmvcv13Config": "mmvcv13Config",
    "mmvcv13Model": "mmvcv13Model",
    "mmvcv15Config": "mmvcv15Config",
    "mmvcv15Model": "mmvcv15Model",

    "soVitsSvc40Config": "soVitsSvc40Config",
    "soVitsSvc40Model": "soVitsSvc40Model",
    "soVitsSvc40Cluster": "soVitsSvc40Cluster",

    "rvcModel": "rvcModel",
    "rvcIndex": "rvcIndex",

    "ddspSvcModel": "ddspSvcModel",
    "ddspSvcModelConfig": "ddspSvcModelConfig",
    "ddspSvcDiffusion": "ddspSvcDiffusion",
    "ddspSvcDiffusionConfig": "ddspSvcDiffusionConfig",

} as const
export type ModelFileKind = typeof ModelFileKind[keyof typeof ModelFileKind]

export type ModelFile = {
    file: File,
    kind: ModelFileKind
    dir: string
}

export type ModelUploadSetting = {
    voiceChangerType: VoiceChangerType,
    slot: number
    isSampleMode: boolean
    sampleId: string | null

    files: ModelFile[]
}
export type ModelFileForServer = Omit<ModelFile, "file"> & {
    name: string,
    kind: ModelFileKind
}
export type ModelUploadSettingForServer = Omit<ModelUploadSetting, "files"> & {
    files: ModelFileForServer[]
}

export type FileUploadSetting = {
    isHalf: boolean
    uploaded: boolean
    defaultTune: number
    defaultIndexRatio: number
    defaultProtect: number
    framework: Framework
    params: string

    mmvcv13Config: ModelData | null
    mmvcv13Model: ModelData | null
    mmvcv15Config: ModelData | null
    mmvcv15Model: ModelData | null
    soVitsSvc40Config: ModelData | null
    soVitsSvc40Model: ModelData | null
    soVitsSvc40Cluster: ModelData | null
    soVitsSvc40v2Config: ModelData | null
    soVitsSvc40v2Model: ModelData | null
    soVitsSvc40v2Cluster: ModelData | null
    rvcModel: ModelData | null
    rvcFeature: ModelData | null
    rvcIndex: ModelData | null

    isSampleMode: boolean
    sampleId: string | null
    rvcIndexDownload: boolean

    ddspSvcModel: ModelData | null
    ddspSvcModelConfig: ModelData | null
    ddspSvcDiffusion: ModelData | null
    ddspSvcDiffusionConfig: ModelData | null

}

export const InitialFileUploadSetting: FileUploadSetting = {
    isHalf: true,
    uploaded: false,
    defaultTune: 0,
    defaultIndexRatio: 1,
    defaultProtect: 0.5,
    framework: Framework.PyTorch,
    params: "{}",

    mmvcv13Config: null,
    mmvcv13Model: null,
    mmvcv15Config: null,
    mmvcv15Model: null,
    soVitsSvc40Config: null,
    soVitsSvc40Model: null,
    soVitsSvc40Cluster: null,
    soVitsSvc40v2Config: null,
    soVitsSvc40v2Model: null,
    soVitsSvc40v2Cluster: null,
    rvcModel: null,
    rvcFeature: null,
    rvcIndex: null,

    isSampleMode: false,
    sampleId: null,
    rvcIndexDownload: true,


    ddspSvcModel: null,
    ddspSvcModelConfig: null,
    ddspSvcDiffusion: null,
    ddspSvcDiffusionConfig: null,
}

type AssetUploadSetting = {
    slot: number
    name: ModelAssetName
    file: string
}

export type UseServerSettingProps = {
    clientType: ClientType | null
    voiceChangerClient: VoiceChangerClient | null
}

export type ServerSettingState = {
    serverSetting: ServerInfo
    updateServerSettings: (setting: ServerInfo) => Promise<void>
    clearSetting: () => Promise<void>
    reloadServerInfo: () => Promise<void>;

    fileUploadSettings: FileUploadSetting[]
    setFileUploadSetting: (slot: number, val: FileUploadSetting) => void
    loadModel: (slot: number) => Promise<void>
    uploadModel: (setting: ModelUploadSetting) => Promise<void>
    uploadProgress: number
    isUploading: boolean

    getOnnx: () => Promise<OnnxExporterInfo>
    mergeModel: (request: MergeModelRequest) => Promise<ServerInfo>
    updateModelDefault: () => Promise<ServerInfo>
    updateModelInfo: (slot: number, key: string, val: string) => Promise<ServerInfo>
    uploadAssets: (slot: number, name: ModelAssetName, file: File) => Promise<void>
}

export const useServerSetting = (props: UseServerSettingProps): ServerSettingState => {
    // const settingRef = useRef<VoiceChangerServerSetting>(DefaultVoiceChangerServerSetting)
    const getDefaultServerSetting = () => {
        if (props.clientType == "MMVCv13") {
            return DefaultServerSetting_MMVCv13
        } else if (props.clientType == "MMVCv15") {
            return DefaultServerSetting_MMVCv15
        } else if (props.clientType == "so-vits-svc-40") {
            return DefaultServerSetting_so_vits_svc_40
        } else if (props.clientType == "so-vits-svc-40_c") {
            console.log("default so_vits_svc_40_c")
            return DefaultServerSetting_so_vits_svc_40_c
        } else if (props.clientType == "so-vits-svc-40v2") {
            return DefaultServerSetting_so_vits_svc_40v2
        } else if (props.clientType == "DDSP-SVC") {
            return DefaultServerSetting_DDSP_SVC
        } else if (props.clientType == "RVC") {
            return DefaultServerSetting_RVC
        } else {
            return DefaultServerSetting_MMVCv15
        }
    }

    const [serverSetting, setServerSetting] = useState<ServerInfo>(getDefaultServerSetting())
    const [fileUploadSettings, setFileUploadSettings] = useState<FileUploadSetting[]>([])
    const { setItem, getItem, removeItem } = useIndexedDB({ clientType: props.clientType })


    // clientTypeが新しく設定されたときに、serverのmodelType動作を変更＋設定反映
    useEffect(() => {
        if (!props.voiceChangerClient) return
        if (!props.clientType) return
        const setInitialSetting = async () => {
            // Set Model Type
            await props.voiceChangerClient!.switchModelType(props.clientType!)

            // Load Default (and Cache) and set
            const defaultServerSetting = getDefaultServerSetting()
            const cachedServerSetting = await getItem(INDEXEDDB_KEY_SERVER)
            let initialSetting: ServerInfo
            if (cachedServerSetting) {
                initialSetting = {
                    ...defaultServerSetting, ...cachedServerSetting as ServerInfo,
                    serverAudioStated: 0,
                    inputSampleRate: 48000
                }// sample rateは時限措置
            } else {
                initialSetting = { ...defaultServerSetting }
            }
            setServerSetting(initialSetting)

            // upload setting
            for (let i = 0; i < Object.values(ServerSettingKey).length; i++) {
                const k = Object.values(ServerSettingKey)[i] as keyof VoiceChangerServerSetting
                const v = initialSetting[k]
                if (v) {
                    props.voiceChangerClient!.updateServerSettings(k, "" + v)
                }
            }

            // Load file upload cache
            const loadedFileUploadSettings: FileUploadSetting[] = []
            for (let i = 0; i < MAX_MODEL_SLOT_NUM; i++) {
                const modleKey = `${INDEXEDDB_KEY_MODEL_DATA}_${i}`
                const fileuploadSetting = await getItem(modleKey)
                if (!fileuploadSetting) {
                    loadedFileUploadSettings.push(InitialFileUploadSetting)
                } else {
                    loadedFileUploadSettings.push(fileuploadSetting as FileUploadSetting)
                }
            }
            setFileUploadSettings(loadedFileUploadSettings)


            reloadServerInfo()
        }

        setInitialSetting()

    }, [props.voiceChangerClient, props.clientType])

    //////////////
    // 設定
    /////////////
    const updateServerSettings = useMemo(() => {
        return async (setting: ServerInfo) => {
            if (!props.voiceChangerClient) return
            for (let i = 0; i < Object.values(ServerSettingKey).length; i++) {
                const k = Object.values(ServerSettingKey)[i] as keyof VoiceChangerServerSetting
                const cur_v = serverSetting[k]
                const new_v = setting[k]
                if (cur_v != new_v) {
                    const res = await props.voiceChangerClient.updateServerSettings(k, "" + new_v)
                    if (res.onnxExecutionProviders && res.onnxExecutionProviders.length > 0) {
                        res.onnxExecutionProvider = res.onnxExecutionProviders[0]
                    } else {
                        res.onnxExecutionProvider = "CPUExecutionProvider"
                    }

                    setServerSetting(res)
                    const storeData = { ...res }
                    storeData.recordIO = 0
                    setItem(INDEXEDDB_KEY_SERVER, storeData)
                }
            }
        }
    }, [props.voiceChangerClient, serverSetting])

    const setFileUploadSetting = useMemo(() => {
        return async (slot: number, fileUploadSetting: FileUploadSetting) => {
            fileUploadSetting.uploaded = false
            fileUploadSettings[slot] = fileUploadSetting
            setFileUploadSettings([...fileUploadSettings])
        }
    }, [fileUploadSettings])


    //////////////
    // 操作
    /////////////
    const [uploadProgress, setUploadProgress] = useState<number>(0)
    const [isUploading, setIsUploading] = useState<boolean>(false)

    // (e) モデルアップロード
    const _uploadFile = useMemo(() => {
        return async (modelData: ModelData, onprogress: (progress: number, end: boolean) => void, dir: string = "") => {
            if (!props.voiceChangerClient) return
            const num = await props.voiceChangerClient.uploadFile(modelData.data!, dir + modelData.filename!, onprogress)
            const res = await props.voiceChangerClient.concatUploadedFile(dir + modelData.filename!, num)
            console.log("uploaded", num, res)
        }
    }, [props.voiceChangerClient])

    const _uploadFile2 = useMemo(() => {
        return async (file: File, onprogress: (progress: number, end: boolean) => void, dir: string = "") => {
            if (!props.voiceChangerClient) return
            console.log("uploading..1.", file)
            console.log("uploading..2.", file.name)
            const num = await props.voiceChangerClient.uploadFile2(dir, file, onprogress)
            const res = await props.voiceChangerClient.concatUploadedFile(dir + file.name, num)
            console.log("uploaded", num, res)
        }
    }, [props.voiceChangerClient])

    // 新しいアップローダ
    const uploadModel = useMemo(() => {
        return async (setting: ModelUploadSetting) => {
            if (!props.voiceChangerClient) {
                return
            }

            setUploadProgress(0)
            setIsUploading(true)


            if (setting.isSampleMode == false) {
                const progRate = 1 / setting.files.length
                for (let i = 0; i < setting.files.length; i++) {
                    const progOffset = 100 * i * progRate
                    await _uploadFile2(setting.files[i].file, (progress: number, _end: boolean) => {
                        setUploadProgress(progress * progRate + progOffset)
                    }, setting.files[i].dir)
                }
            }
            const params: ModelUploadSettingForServer = {
                ...setting, files: setting.files.map((f) => { return { name: f.file.name, kind: f.kind, dir: f.dir } })
            }

            const loadPromise = props.voiceChangerClient.loadModel(
                0,
                false,
                JSON.stringify(params),
            )
            await loadPromise

            setUploadProgress(0)
            setIsUploading(false)
            reloadServerInfo()

        }
    }, [props.voiceChangerClient])


    // 古いアップローダ（新GUIへ以降まで、当分残しておく。）
    const loadModel = useMemo(() => {
        return async (slot: number) => {
            const fileUploadSetting = fileUploadSettings[slot]
            console.log("[loadModel]", fileUploadSetting)
            console.log("[loadModel] model:", props.clientType)

            if (fileUploadSetting.isSampleMode == false) {
                if (props.clientType == "MMVCv13") {
                    if (!fileUploadSetting.mmvcv13Config) {
                        throw new ModelLoadException("Config")
                    }
                    if (!fileUploadSetting.mmvcv13Model) {
                        throw new ModelLoadException("Model")
                    }
                } else if (props.clientType == "MMVCv15") {
                    if (!fileUploadSetting.mmvcv15Config) {
                        throw new ModelLoadException("Config")
                    }
                    if (!fileUploadSetting.mmvcv15Model) {
                        throw new ModelLoadException("Model")
                    }
                } else if (props.clientType == "so-vits-svc-40") {
                    if (!fileUploadSetting.soVitsSvc40Config) {
                        throw new ModelLoadException("Config")
                    }
                    if (!fileUploadSetting.soVitsSvc40Model) {
                        throw new ModelLoadException("Model")
                    }
                } else if (props.clientType == "so-vits-svc-40v2") {
                    if (!fileUploadSetting.soVitsSvc40v2Config) {
                        throw new ModelLoadException("Config")
                    }
                    if (!fileUploadSetting.soVitsSvc40v2Model) {
                        throw new ModelLoadException("Model")
                    }
                } else if (props.clientType == "RVC") {
                    if (!fileUploadSetting.rvcModel) {
                        throw new ModelLoadException("Model")
                    }
                } else if (props.clientType == "DDSP-SVC") {
                    if (!fileUploadSetting.ddspSvcModel) {
                        throw new ModelLoadException("DDSP-Model")
                    }
                    if (!fileUploadSetting.ddspSvcModelConfig) {
                        throw new ModelLoadException("DDSP-Config")
                    }
                    if (!fileUploadSetting.ddspSvcDiffusion) {
                        throw new ModelLoadException("Diff-Model")
                    }
                    if (!fileUploadSetting.ddspSvcDiffusionConfig) {
                        throw new ModelLoadException("Diff-Config")
                    }
                } else {
                }
            } else {//Sampleモード
                if (!fileUploadSetting.sampleId) {
                    throw new ModelLoadException("SampleId")
                }
            }


            if (!props.voiceChangerClient) return

            setUploadProgress(0)
            setIsUploading(true)

            // normal models(MMVC13,15, so-vits-svc, RVC)
            const normalModels = [
                fileUploadSetting.mmvcv13Config,
                fileUploadSetting.mmvcv13Model,
                fileUploadSetting.mmvcv15Config,
                fileUploadSetting.mmvcv15Model,
                fileUploadSetting.soVitsSvc40Config,
                fileUploadSetting.soVitsSvc40Model,
                fileUploadSetting.soVitsSvc40Cluster,
                fileUploadSetting.soVitsSvc40v2Config,
                fileUploadSetting.soVitsSvc40v2Model,
                fileUploadSetting.soVitsSvc40v2Cluster,

            ].filter(x => { return x != null }) as ModelData[]
            console.log("[SENDING FILE]", normalModels)
            for (let i = 0; i < normalModels.length; i++) {
                if (!normalModels[i].data) {
                    // const fileSize = normalModels[i].file!.size / 1024 / 1024
                    normalModels[i].data = await normalModels[i].file!.arrayBuffer()
                    normalModels[i].filename = await normalModels[i].file!.name
                }
            }
            if (fileUploadSetting.isSampleMode == false) {
                for (let i = 0; i < normalModels.length; i++) {
                    const progRate = 1 / normalModels.length
                    const progOffset = 100 * i * progRate
                    await _uploadFile(normalModels[i], (progress: number, _end: boolean) => {
                        setUploadProgress(progress * progRate + progOffset)
                    })
                    // await _uploadFile2(normalModels[i].file!, (progress: number, _end: boolean) => {
                    //     setUploadProgress(progress * progRate + progOffset)
                    // })
                }
            }

            // slotModel ローカルキャッシュ無効(RVC)
            const slotModels = [
                fileUploadSetting.rvcModel,
                fileUploadSetting.rvcIndex,

            ].filter(x => { return x != null }) as ModelData[]
            for (let i = 0; i < slotModels.length; i++) {
                if (!slotModels[i].data) {
                    slotModels[i].filename = await slotModels[i].file!.name
                }
            }
            if (fileUploadSetting.isSampleMode == false) {
                for (let i = 0; i < slotModels.length; i++) {
                    const progRate = 1 / slotModels.length
                    const progOffset = 100 * i * progRate
                    await _uploadFile2(slotModels[i].file!, (progress: number, _end: boolean) => {
                        setUploadProgress(progress * progRate + progOffset)
                    })
                }
            }



            // DDSP-SVC (ファイル名（config）が被る可能性があるため、アップロードフォルダを分ける必要がある)
            const ddspSvcModels = [fileUploadSetting.ddspSvcModel, fileUploadSetting.ddspSvcModelConfig, fileUploadSetting.ddspSvcDiffusion, fileUploadSetting.ddspSvcDiffusionConfig].filter(x => { return x != null }) as ModelData[]
            for (let i = 0; i < ddspSvcModels.length; i++) {
                if (!ddspSvcModels[i].data) {
                    ddspSvcModels[i].data = await ddspSvcModels[i].file!.arrayBuffer()
                    ddspSvcModels[i].filename = await ddspSvcModels[i].file!.name
                }
            }
            if (fileUploadSetting.isSampleMode == false) {
                for (let i = 0; i < ddspSvcModels.length; i++) {
                    const progRate = 1 / ddspSvcModels.length
                    const progOffset = 100 * i * progRate
                    const dir = i == 0 || i == 1 ? "ddsp_mod/" : "ddsp_diff/"
                    await _uploadFile(ddspSvcModels[i], (progress: number, _end: boolean) => {
                        setUploadProgress(progress * progRate + progOffset)
                    }, dir)
                }
            }

            // const configFileName = fileUploadSetting.configFile?.filename || "-"
            const params = JSON.stringify({
                defaultTune: fileUploadSetting.defaultTune || 0,
                defaultIndexRatio: fileUploadSetting.defaultIndexRatio || 1,
                defaultProtect: fileUploadSetting.defaultProtect || 0.5,
                sampleId: fileUploadSetting.isSampleMode ? fileUploadSetting.sampleId || "" : "",
                rvcIndexDownload: fileUploadSetting.rvcIndexDownload || false,
                files: fileUploadSetting.isSampleMode ? {} : {
                    mmvcv13Config: props.clientType == "MMVCv13" ? fileUploadSetting.mmvcv13Config?.filename || "" : "",
                    mmvcv13Model: props.clientType == "MMVCv13" ? fileUploadSetting.mmvcv13Model?.filename || "" : "",
                    mmvcv15Config: props.clientType == "MMVCv15" ? fileUploadSetting.mmvcv15Config?.filename || "" : "",
                    mmvcv15Model: props.clientType == "MMVCv15" ? fileUploadSetting.mmvcv15Model?.filename || "" : "",
                    soVitsSvc40Config: props.clientType == "so-vits-svc-40" ? fileUploadSetting.soVitsSvc40Config?.filename || "" : "",
                    soVitsSvc40Model: props.clientType == "so-vits-svc-40" ? fileUploadSetting.soVitsSvc40Model?.filename || "" : "",
                    soVitsSvc40Cluster: props.clientType == "so-vits-svc-40" ? fileUploadSetting.soVitsSvc40Cluster?.filename || "" : "",
                    rvcModel: props.clientType == "RVC" ? fileUploadSetting.rvcModel?.filename || "" : "",
                    rvcIndex: props.clientType == "RVC" ? fileUploadSetting.rvcIndex?.filename || "" : "",
                    rvcFeature: props.clientType == "RVC" ? fileUploadSetting.rvcFeature?.filename || "" : "",

                    ddspSvcModel: props.clientType == "DDSP-SVC" ? fileUploadSetting.ddspSvcModel?.filename ? "ddsp_mod/" + fileUploadSetting.ddspSvcModel?.filename : "" : "",
                    ddspSvcModelConfig: props.clientType == "DDSP-SVC" ? fileUploadSetting.ddspSvcModelConfig?.filename ? "ddsp_mod/" + fileUploadSetting.ddspSvcModelConfig?.filename : "" : "",
                    ddspSvcDiffusion: props.clientType == "DDSP-SVC" ? fileUploadSetting.ddspSvcDiffusion?.filename ? "ddsp_diff/" + fileUploadSetting.ddspSvcDiffusion?.filename : "" : "",
                    ddspSvcDiffusionConfig: props.clientType == "DDSP-SVC" ? fileUploadSetting.ddspSvcDiffusionConfig?.filename ? "ddsp_diff/" + fileUploadSetting.ddspSvcDiffusionConfig.filename : "" : "",
                }
            })

            if (fileUploadSetting.isHalf == undefined) {
                fileUploadSetting.isHalf = false
            }

            console.log("PARAMS:", params)

            const loadPromise = props.voiceChangerClient.loadModel(
                slot,
                fileUploadSetting.isHalf,
                params,
            )


            // サーバでロード中にキャッシュにセーブ
            storeToCache(slot, fileUploadSetting)
            await loadPromise

            fileUploadSetting.uploaded = true
            fileUploadSettings[slot] = fileUploadSetting
            setFileUploadSettings([...fileUploadSettings])


            setUploadProgress(0)
            setIsUploading(false)
            reloadServerInfo()
        }
    }, [fileUploadSettings, props.voiceChangerClient, props.clientType])


    const storeToCache = (slot: number, fileUploadSetting: FileUploadSetting) => {
        try {
            const saveData: FileUploadSetting = {
                isHalf: fileUploadSetting.isHalf, // キャッシュとしては不使用。guiで上書きされる。
                uploaded: false, // キャッシュから読み込まれるときには、まだuploadされていないから。
                defaultTune: fileUploadSetting.defaultTune,
                defaultIndexRatio: fileUploadSetting.defaultIndexRatio,
                defaultProtect: fileUploadSetting.defaultProtect,
                framework: fileUploadSetting.framework,
                params: fileUploadSetting.params,

                mmvcv13Config: fileUploadSetting.mmvcv13Config ? { data: fileUploadSetting.mmvcv13Config.data, filename: fileUploadSetting.mmvcv13Config.filename } : null,
                mmvcv13Model: fileUploadSetting.mmvcv13Model ? { data: fileUploadSetting.mmvcv13Model.data, filename: fileUploadSetting.mmvcv13Model.filename } : null,
                mmvcv15Config: fileUploadSetting.mmvcv15Config ? { data: fileUploadSetting.mmvcv15Config.data, filename: fileUploadSetting.mmvcv15Config.filename } : null,
                mmvcv15Model: fileUploadSetting.mmvcv15Model ? { data: fileUploadSetting.mmvcv15Model.data, filename: fileUploadSetting.mmvcv15Model.filename } : null,
                soVitsSvc40Config: fileUploadSetting.soVitsSvc40Config ? { data: fileUploadSetting.soVitsSvc40Config.data, filename: fileUploadSetting.soVitsSvc40Config.filename } : null,
                soVitsSvc40Model: fileUploadSetting.soVitsSvc40Model ? { data: fileUploadSetting.soVitsSvc40Model.data, filename: fileUploadSetting.soVitsSvc40Model.filename } : null,
                soVitsSvc40Cluster: fileUploadSetting.soVitsSvc40Cluster ? { data: fileUploadSetting.soVitsSvc40Cluster.data, filename: fileUploadSetting.soVitsSvc40Cluster.filename } : null,
                soVitsSvc40v2Config: fileUploadSetting.soVitsSvc40v2Config ? { data: fileUploadSetting.soVitsSvc40v2Config.data, filename: fileUploadSetting.soVitsSvc40v2Config.filename } : null,
                soVitsSvc40v2Model: fileUploadSetting.soVitsSvc40v2Model ? { data: fileUploadSetting.soVitsSvc40v2Model.data, filename: fileUploadSetting.soVitsSvc40v2Model.filename } : null,
                soVitsSvc40v2Cluster: fileUploadSetting.soVitsSvc40v2Cluster ? { data: fileUploadSetting.soVitsSvc40v2Cluster.data, filename: fileUploadSetting.soVitsSvc40v2Cluster.filename } : null,
                rvcModel: fileUploadSetting.rvcModel ? { data: fileUploadSetting.rvcModel.data, filename: fileUploadSetting.rvcModel.filename } : null,
                rvcIndex: fileUploadSetting.rvcIndex ? { data: fileUploadSetting.rvcIndex.data, filename: fileUploadSetting.rvcIndex.filename } : null,
                rvcFeature: fileUploadSetting.rvcFeature ? { data: fileUploadSetting.rvcFeature.data, filename: fileUploadSetting.rvcFeature.filename } : null,

                ddspSvcModel: fileUploadSetting.ddspSvcModel ? { data: fileUploadSetting.ddspSvcModel.data, filename: fileUploadSetting.ddspSvcModel.filename } : null,
                ddspSvcModelConfig: fileUploadSetting.ddspSvcModelConfig ? { data: fileUploadSetting.ddspSvcModelConfig.data, filename: fileUploadSetting.ddspSvcModelConfig.filename } : null,
                ddspSvcDiffusion: fileUploadSetting.ddspSvcDiffusion ? { data: fileUploadSetting.ddspSvcDiffusion.data, filename: fileUploadSetting.ddspSvcDiffusion.filename } : null,
                ddspSvcDiffusionConfig: fileUploadSetting.ddspSvcDiffusionConfig ? { data: fileUploadSetting.ddspSvcDiffusionConfig.data, filename: fileUploadSetting.ddspSvcDiffusionConfig.filename } : null,

                isSampleMode: fileUploadSetting.isSampleMode,
                sampleId: fileUploadSetting.sampleId,
                rvcIndexDownload: fileUploadSetting.rvcIndexDownload,
            }
            setItem(`${INDEXEDDB_KEY_MODEL_DATA}_${slot}`, saveData)
        } catch (e) {
            console.log("Excpetion:::::::::", e)
        }
    }



    const uploadAssets = useMemo(() => {
        return async (slot: number, name: ModelAssetName, file: File) => {
            if (!props.voiceChangerClient) return

            await _uploadFile2(file, (progress: number, _end: boolean) => {
                console.log(progress, _end)
            })
            const assetUploadSetting: AssetUploadSetting = {
                slot,
                name,
                file: file.name
            }
            await props.voiceChangerClient.uploadAssets(JSON.stringify(assetUploadSetting))
            reloadServerInfo()
        }
    }, [fileUploadSettings, props.voiceChangerClient, props.clientType])



    const reloadServerInfo = useMemo(() => {
        return async () => {

            if (!props.voiceChangerClient) return
            const res = await props.voiceChangerClient.getServerSettings()
            setServerSetting(res)
            const storeData = { ...res }
            storeData.recordIO = 0
            setItem(INDEXEDDB_KEY_SERVER, storeData)
        }
    }, [props.voiceChangerClient])

    const clearSetting = async () => {
        await removeItem(INDEXEDDB_KEY_SERVER)
        await removeItem(INDEXEDDB_KEY_MODEL_DATA)
        for (let i = 0; i < MAX_MODEL_SLOT_NUM; i++) {
            const modleKey = `${INDEXEDDB_KEY_MODEL_DATA}_${i}`
            await removeItem(modleKey)
        }
    }


    const getOnnx = async () => {
        return props.voiceChangerClient!.getOnnx()
    }

    const mergeModel = async (request: MergeModelRequest) => {
        const serverInfo = await props.voiceChangerClient!.mergeModel(request)
        setServerSetting(serverInfo)
        return serverInfo
    }

    const updateModelDefault = async () => {
        const serverInfo = await props.voiceChangerClient!.updateModelDefault()
        setServerSetting(serverInfo)
        return serverInfo
    }
    const updateModelInfo = async (slot: number, key: string, val: string) => {
        const serverInfo = await props.voiceChangerClient!.updateModelInfo(slot, key, val)
        setServerSetting(serverInfo)
        return serverInfo
    }

    return {
        serverSetting,
        updateServerSettings,
        clearSetting,
        reloadServerInfo,

        fileUploadSettings,
        setFileUploadSetting,
        loadModel,
        uploadModel,
        uploadProgress,
        isUploading,
        getOnnx,
        mergeModel,
        updateModelDefault,
        updateModelInfo,
        uploadAssets
    }
}