import { useState, useMemo, useEffect } from "react"
import { VoiceChangerServerSetting, ServerInfo, ServerSettingKey, INDEXEDDB_KEY_SERVER, INDEXEDDB_KEY_MODEL_DATA, ClientType, DefaultServerSetting_MMVCv13, DefaultServerSetting_MMVCv15, DefaultServerSetting_so_vits_svc_40, DefaultServerSetting_RVC, OnnxExporterInfo, DefaultServerSetting_DDSP_SVC, MAX_MODEL_SLOT_NUM, Framework, MergeModelRequest, VoiceChangerType } from "../const"
import { VoiceChangerClient } from "../VoiceChangerClient"
import { useIndexedDB } from "./useIndexedDB"


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
    params: any
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