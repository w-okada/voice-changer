import { useState, useMemo, useEffect } from "react"
import { VoiceChangerServerSetting, ServerInfo, ServerSettingKey, INDEXEDDB_KEY_SERVER, INDEXEDDB_KEY_MODEL_DATA, ClientType, DefaultServerSetting_MMVCv13, DefaultServerSetting_MMVCv15, DefaultServerSetting_so_vits_svc_40v2 } from "../const"
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
    hubertTorchModel: ModelData | null // !! 注意!! hubertTorchModelは固定値で上書きされるため、設定しても効果ない。
}

const InitialFileUploadSetting: FileUploadSetting = {
    pyTorchModel: null,
    configFile: null,
    onnxModel: null,
    clusterTorchModel: null,
    hubertTorchModel: null,
}

export type UseServerSettingProps = {
    clientType: ClientType
    voiceChangerClient: VoiceChangerClient | null
}

export type ServerSettingState = {
    serverSetting: ServerInfo
    updateServerSettings: (setting: ServerInfo) => Promise<void>
    clearSetting: () => Promise<void>
    reloadServerInfo: () => Promise<void>;

    fileUploadSetting: FileUploadSetting
    setFileUploadSetting: (val: FileUploadSetting) => void
    loadModel: () => Promise<void>
    uploadProgress: number
    isUploading: boolean

}

export const useServerSetting = (props: UseServerSettingProps): ServerSettingState => {
    // const settingRef = useRef<VoiceChangerServerSetting>(DefaultVoiceChangerServerSetting)
    const defaultServerSetting = useMemo(() => {
        if (props.clientType == "MMVCv13") {
            return DefaultServerSetting_MMVCv13
        } else if (props.clientType == "MMVCv15") {
            return DefaultServerSetting_MMVCv15
        } else if (props.clientType == "so_vits_svc_40v2" || props.clientType == "so_vits_svc_40v2_tsukuyomi") {
            return DefaultServerSetting_so_vits_svc_40v2
        } else {
            return DefaultServerSetting_MMVCv15
        }
    }, [])
    const [serverSetting, setServerSetting] = useState<ServerInfo>(defaultServerSetting)
    const [fileUploadSetting, setFileUploadSetting] = useState<FileUploadSetting>(InitialFileUploadSetting)
    const { setItem, getItem, removeItem } = useIndexedDB({ clientType: props.clientType })


    // DBから設定取得（キャッシュによる初期化）
    useEffect(() => {
        const loadCache = async () => {
            const setting = await getItem(INDEXEDDB_KEY_SERVER)
            if (!setting) {
            } else {
                setServerSetting(setting as ServerInfo)
            }

            const fileuploadSetting = await getItem(INDEXEDDB_KEY_MODEL_DATA)
            if (!fileuploadSetting) {
            } else {
                setFileUploadSetting(fileuploadSetting as FileUploadSetting)
            }
        }

        loadCache()
    }, [])

    // クライアントへ設定反映 (キャッシュ反映)
    useEffect(() => {
        if (!props.voiceChangerClient) return
        for (let i = 0; i < Object.values(ServerSettingKey).length; i++) {
            const k = Object.values(ServerSettingKey)[i] as keyof VoiceChangerServerSetting
            const v = serverSetting[k]
            if (v) {
                props.voiceChangerClient.updateServerSettings(k, "" + v)
            }
        }
        reloadServerInfo()
    }, [props.voiceChangerClient])

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
                    if (res.onnxExecutionProviders.length > 0) {
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
        return async () => {
            if (!fileUploadSetting.pyTorchModel && !fileUploadSetting.onnxModel) {
                alert("PyTorchモデルとONNXモデルのどちらか一つ以上指定する必要があります。")
                return
            }
            if (!fileUploadSetting.configFile) {
                alert("Configファイルを指定する必要があります。")
                return
            }

            // if (props.clientType == "so_vits_svc_40v2c" && !fileUploadSetting.hubertTorchModel) {
            //     alert("content vecのファイルを指定する必要があります。")
            //     return
            // }
            if (!props.voiceChangerClient) return

            setUploadProgress(0)
            setIsUploading(true)

            // ファイルをメモリにロード
            if (fileUploadSetting.onnxModel && !fileUploadSetting.onnxModel.data) {
                fileUploadSetting.onnxModel.data = await fileUploadSetting.onnxModel.file!.arrayBuffer()
                fileUploadSetting.onnxModel.filename = await fileUploadSetting.onnxModel.file!.name
            }
            if (fileUploadSetting.pyTorchModel && !fileUploadSetting.pyTorchModel.data) {
                fileUploadSetting.pyTorchModel.data = await fileUploadSetting.pyTorchModel.file!.arrayBuffer()
                fileUploadSetting.pyTorchModel.filename = await fileUploadSetting.pyTorchModel.file!.name
            }
            if (!fileUploadSetting.configFile.data) {
                fileUploadSetting.configFile.data = await fileUploadSetting.configFile.file!.arrayBuffer()
                fileUploadSetting.configFile.filename = await fileUploadSetting.configFile.file!.name
            }
            // if (props.clientType == "so_vits_svc_40v2c" && !fileUploadSetting.hubertTorchModel!.data) {
            //     fileUploadSetting.hubertTorchModel!.data = await fileUploadSetting.hubertTorchModel!.file!.arrayBuffer()
            //     fileUploadSetting.hubertTorchModel!.filename = await fileUploadSetting.hubertTorchModel!.file!.name
            // }
            if (fileUploadSetting.clusterTorchModel) {
                if (props.clientType == "so_vits_svc_40v2" && !fileUploadSetting.clusterTorchModel!.data) {
                    fileUploadSetting.clusterTorchModel!.data = await fileUploadSetting.clusterTorchModel!.file!.arrayBuffer()
                    fileUploadSetting.clusterTorchModel!.filename = await fileUploadSetting.clusterTorchModel!.file!.name
                }
            }

            // ファイルをサーバにアップロード
            const models = [fileUploadSetting.onnxModel, fileUploadSetting.pyTorchModel, fileUploadSetting.clusterTorchModel /*, fileUploadSetting.hubertTorchModel*/].filter(x => { return x != null }) as ModelData[]
            for (let i = 0; i < models.length; i++) {
                const progRate = 1 / models.length
                const progOffset = 100 * i * progRate
                await _uploadFile(models[i], (progress: number, _end: boolean) => {
                    // console.log(progress * progRate + progOffset, end, progRate,)
                    setUploadProgress(progress * progRate + progOffset)
                })
            }

            await _uploadFile(fileUploadSetting.configFile, (progress: number, end: boolean) => {
                console.log(progress, end)
            })

            // !! 注意!! hubertTorchModelは固定値で上書きされるため、設定しても効果ない。
            const loadPromise = props.voiceChangerClient.loadModel(fileUploadSetting.configFile.filename!, fileUploadSetting.pyTorchModel?.filename || null, fileUploadSetting.onnxModel?.filename || null, fileUploadSetting.clusterTorchModel?.filename || null, fileUploadSetting.hubertTorchModel?.filename || null)

            // サーバでロード中にキャッシュにセーブ
            try {
                const saveData: FileUploadSetting = {
                    pyTorchModel: fileUploadSetting.pyTorchModel ? { data: fileUploadSetting.pyTorchModel.data, filename: fileUploadSetting.pyTorchModel.filename } : null,
                    onnxModel: fileUploadSetting.onnxModel ? { data: fileUploadSetting.onnxModel.data, filename: fileUploadSetting.onnxModel.filename } : null,
                    configFile: { data: fileUploadSetting.configFile.data, filename: fileUploadSetting.configFile.filename },
                    hubertTorchModel: fileUploadSetting.hubertTorchModel ? {
                        data: fileUploadSetting.hubertTorchModel.data, filename: fileUploadSetting.hubertTorchModel.filename
                    } : null,
                    clusterTorchModel: fileUploadSetting.clusterTorchModel ? {
                        data: fileUploadSetting.clusterTorchModel.data, filename: fileUploadSetting.clusterTorchModel.filename
                    } : null
                }
                setItem(INDEXEDDB_KEY_MODEL_DATA, saveData)

            } catch (e) {
                console.log("Excpetion:::::::::", e)
            }

            await loadPromise
            setUploadProgress(0)
            setIsUploading(false)
            reloadServerInfo()
        }
    }, [fileUploadSetting, props.voiceChangerClient, props.clientType])

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
    }


    return {
        serverSetting,
        updateServerSettings,
        clearSetting,
        reloadServerInfo,

        fileUploadSetting,
        setFileUploadSetting,
        loadModel,
        uploadProgress,
        isUploading,
    }
}