import { useState, useMemo, useEffect } from "react"
import { VoiceChangerServerSetting, ServerInfo, ServerSettingKey, INDEXEDDB_KEY_SERVER, INDEXEDDB_KEY_MODEL_DATA, ClientType, DefaultServerSetting_MMVCv13, DefaultServerSetting_MMVCv15, DefaultServerSetting_so_vits_svc_40v2, DefaultServerSetting_so_vits_svc_40, DefaultServerSetting_so_vits_svc_40_c, DefaultServerSetting_RVC, OnnxExporterInfo, DefaultServerSetting_DDSP_SVC, MAX_MODEL_SLOT_NUM, Framework } from "../const"
import { VoiceChangerClient } from "../VoiceChangerClient"
import { useIndexedDB } from "./useIndexedDB"


type ModelData = {
    file?: File
    data?: ArrayBuffer
    filename?: string
}

export type FileUploadSetting = {
    pyTorchModel: ModelData | null
    onnxModel: ModelData | null
    configFile: ModelData | null
    clusterTorchModel: ModelData | null

    feature: ModelData | null //RVC
    index: ModelData | null   //RVC

    isHalf: boolean
    uploaded: boolean
    defaultTune: number
    framework: Framework
    params: string

}

const InitialFileUploadSetting: FileUploadSetting = {
    pyTorchModel: null,
    configFile: null,
    onnxModel: null,
    clusterTorchModel: null,

    feature: null,
    index: null,

    isHalf: true,
    uploaded: false,
    defaultTune: 0,
    framework: Framework.PyTorch,
    params: "{}"
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
    uploadProgress: number
    isUploading: boolean

    getOnnx: () => Promise<OnnxExporterInfo>
    // updateDefaultTune: (slot: number, tune: number) => void

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
                initialSetting = { ...defaultServerSetting, ...cachedServerSetting as ServerInfo, inputSampleRate: 48000 }// sample rateは時限措置
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
        return async (modelData: ModelData, onprogress: (progress: number, end: boolean) => void) => {
            if (!props.voiceChangerClient) return
            const num = await props.voiceChangerClient.uploadFile(modelData.data!, modelData.filename!, onprogress)
            const res = await props.voiceChangerClient.concatUploadedFile(modelData.filename!, num)
            console.log("uploaded", num, res)
        }
    }, [props.voiceChangerClient])
    const loadModel = useMemo(() => {
        return async (slot: number) => {
            if (!fileUploadSettings[slot].pyTorchModel && !fileUploadSettings[slot].onnxModel) {
                alert("PyTorchモデルとONNXモデルのどちらか一つ以上指定する必要があります。")
                return
            }
            if (!fileUploadSettings[slot].configFile && props.clientType != "RVC") {
                alert("Configファイルを指定する必要があります。")
                return
            }

            if (!props.voiceChangerClient) return

            setUploadProgress(0)
            setIsUploading(true)

            // ファイルをメモリにロード(dataがある場合は、キャッシュから読まれていると想定しスキップ)
            const fileUploadSetting = fileUploadSettings[slot]
            if (fileUploadSetting.onnxModel && !fileUploadSetting.onnxModel.data) {
                fileUploadSetting.onnxModel.data = await fileUploadSetting.onnxModel.file!.arrayBuffer()
                fileUploadSetting.onnxModel.filename = await fileUploadSetting.onnxModel.file!.name
            }
            if (fileUploadSetting.pyTorchModel && !fileUploadSetting.pyTorchModel.data) {
                fileUploadSetting.pyTorchModel.data = await fileUploadSetting.pyTorchModel.file!.arrayBuffer()
                fileUploadSetting.pyTorchModel.filename = await fileUploadSetting.pyTorchModel.file!.name
            }
            if (fileUploadSetting.configFile && !fileUploadSetting.configFile.data) {
                fileUploadSetting.configFile.data = await fileUploadSetting.configFile.file!.arrayBuffer()
                fileUploadSetting.configFile.filename = await fileUploadSetting.configFile.file!.name
            }

            if (fileUploadSetting.clusterTorchModel) {
                if ((props.clientType == "so-vits-svc-40v2" || props.clientType == "so-vits-svc-40") && !fileUploadSetting.clusterTorchModel!.data) {
                    fileUploadSetting.clusterTorchModel!.data = await fileUploadSetting.clusterTorchModel!.file!.arrayBuffer()
                    fileUploadSetting.clusterTorchModel!.filename = await fileUploadSetting.clusterTorchModel!.file!.name
                }
            }

            if (fileUploadSetting.feature) {
                if ((props.clientType == "RVC") && !fileUploadSetting.feature!.data) {
                    fileUploadSetting.feature!.data = await fileUploadSetting.feature!.file!.arrayBuffer()
                    fileUploadSetting.feature!.filename = await fileUploadSetting.feature!.file!.name
                }
            }
            if (fileUploadSetting.index) {
                if ((props.clientType == "RVC") && !fileUploadSetting.index!.data) {
                    fileUploadSetting.index!.data = await fileUploadSetting.index!.file!.arrayBuffer()
                    fileUploadSetting.index!.filename = await fileUploadSetting.index!.file!.name
                }
            }

            // ファイルをサーバにアップロード
            const models = [
                fileUploadSetting.onnxModel,
                fileUploadSetting.pyTorchModel,
                fileUploadSetting.clusterTorchModel,
                fileUploadSetting.feature,
                fileUploadSetting.index,
            ].filter(x => { return x != null }) as ModelData[]
            for (let i = 0; i < models.length; i++) {
                const progRate = 1 / models.length
                const progOffset = 100 * i * progRate
                await _uploadFile(models[i], (progress: number, _end: boolean) => {
                    // console.log(progress * progRate + progOffset, end, progRate,)
                    setUploadProgress(progress * progRate + progOffset)
                })
            }

            if (fileUploadSetting.configFile) {
                await _uploadFile(fileUploadSetting.configFile, (progress: number, end: boolean) => {
                    console.log(progress, end)
                })
            }

            const configFileName = fileUploadSetting.configFile ? fileUploadSetting.configFile.filename || "-" : "-"
            const params = JSON.stringify({
                trans: fileUploadSetting.defaultTune || 0
            })
            if (fileUploadSetting.isHalf == undefined) {
                fileUploadSetting.isHalf = false
            }
            const loadPromise = props.voiceChangerClient.loadModel(
                slot,
                configFileName,
                fileUploadSetting.pyTorchModel?.filename || null,
                fileUploadSetting.onnxModel?.filename || null,
                fileUploadSetting.clusterTorchModel?.filename || null,
                fileUploadSetting.feature?.filename || null,
                fileUploadSetting.index?.filename || null,
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


    // const updateDefaultTune = (slot: number, tune: number) => {
    //     fileUploadSettings[slot].defaultTune = tune
    //     storeToCache(slot, fileUploadSettings[slot])
    //     setFileUploadSettings([...fileUploadSettings])
    // }

    const storeToCache = (slot: number, fileUploadSetting: FileUploadSetting) => {
        try {
            const saveData: FileUploadSetting = {
                pyTorchModel: fileUploadSetting.pyTorchModel ? { data: fileUploadSetting.pyTorchModel.data, filename: fileUploadSetting.pyTorchModel.filename } : null,
                onnxModel: fileUploadSetting.onnxModel ? { data: fileUploadSetting.onnxModel.data, filename: fileUploadSetting.onnxModel.filename } : null,
                configFile: fileUploadSetting.configFile ? { data: fileUploadSetting.configFile.data, filename: fileUploadSetting.configFile.filename } : null,
                clusterTorchModel: fileUploadSetting.clusterTorchModel ? {
                    data: fileUploadSetting.clusterTorchModel.data, filename: fileUploadSetting.clusterTorchModel.filename
                } : null,
                feature: fileUploadSetting.feature ? {
                    data: fileUploadSetting.feature.data, filename: fileUploadSetting.feature.filename
                } : null,
                index: fileUploadSetting.index ? {
                    data: fileUploadSetting.index.data, filename: fileUploadSetting.index.filename
                } : null,
                isHalf: fileUploadSetting.isHalf, // キャッシュとしては不使用。guiで上書きされる。
                uploaded: false, // キャッシュから読み込まれるときには、まだuploadされていないから。
                defaultTune: fileUploadSetting.defaultTune,
                framework: fileUploadSetting.framework,
                params: fileUploadSetting.params
            }
            setItem(`${INDEXEDDB_KEY_MODEL_DATA}_${slot}`, saveData)
        } catch (e) {
            console.log("Excpetion:::::::::", e)
        }
    }



    const reloadServerInfo = useMemo(() => {
        return async () => {
            console.log("reload server info")

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

    return {
        serverSetting,
        updateServerSettings,
        clearSetting,
        reloadServerInfo,

        fileUploadSettings,
        setFileUploadSetting,
        loadModel,
        uploadProgress,
        isUploading,
        getOnnx,
        // updateDefaultTune,
    }
}