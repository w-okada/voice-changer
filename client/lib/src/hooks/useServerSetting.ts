import { useState, useMemo, useEffect } from "react"
import { VoiceChangerServerSetting, ServerInfo, ServerSettingKey, INDEXEDDB_KEY_SERVER, INDEXEDDB_KEY_MODEL_DATA, ClientType, DefaultServerSetting_MMVCv13, DefaultServerSetting_MMVCv15, DefaultServerSetting_so_vits_svc_40v2, DefaultServerSetting_so_vits_svc_40, DefaultServerSetting_so_vits_svc_40_c, DefaultServerSetting_RVC, OnnxExporterInfo } from "../const"
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

    feature: ModelData | null //RVC
    index: ModelData | null   //RVC

    isHalf: boolean

}

const InitialFileUploadSetting: FileUploadSetting = {
    pyTorchModel: null,
    configFile: null,
    onnxModel: null,
    clusterTorchModel: null,
    hubertTorchModel: null,

    feature: null,
    index: null,

    isHalf: true
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

    fileUploadSetting: FileUploadSetting
    setFileUploadSetting: (val: FileUploadSetting) => void
    loadModel: () => Promise<void>
    uploadProgress: number
    isUploading: boolean

    getOnnx: () => Promise<OnnxExporterInfo>

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
        } else if (props.clientType == "RVC") {
            return DefaultServerSetting_RVC
        } else {
            return DefaultServerSetting_MMVCv15
        }
    }

    const [serverSetting, setServerSetting] = useState<ServerInfo>(getDefaultServerSetting())
    const [fileUploadSetting, setFileUploadSetting] = useState<FileUploadSetting>(InitialFileUploadSetting)
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
                initialSetting = { ...defaultServerSetting, ...cachedServerSetting as ServerInfo }
                console.log("Initial Setting1:", initialSetting)
            } else {
                initialSetting = { ...defaultServerSetting }
                console.log("Initial Setting2:", initialSetting)
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
            const fileuploadSetting = await getItem(INDEXEDDB_KEY_MODEL_DATA)
            if (!fileuploadSetting) {
            } else {
                setFileUploadSetting(fileuploadSetting as FileUploadSetting)
            }

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
            if (!fileUploadSetting.configFile && props.clientType != "RVC") {
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

            // ファイルをメモリにロード(dataがある場合は、キャッシュから読まれていると想定しスキップ)
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
            // if (props.clientType == "so_vits_svc_40v2c" && !fileUploadSetting.hubertTorchModel!.data) {
            //     fileUploadSetting.hubertTorchModel!.data = await fileUploadSetting.hubertTorchModel!.file!.arrayBuffer()
            //     fileUploadSetting.hubertTorchModel!.filename = await fileUploadSetting.hubertTorchModel!.file!.name
            // }
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

            // !! 注意!! hubertTorchModelは固定値で上書きされるため、設定しても効果ない。
            const configFileName = fileUploadSetting.configFile ? fileUploadSetting.configFile.filename || "-" : "-"
            console.log("IS HALF", fileUploadSetting.isHalf)
            const loadPromise = props.voiceChangerClient.loadModel(
                0,
                configFileName,
                fileUploadSetting.pyTorchModel?.filename || null,
                fileUploadSetting.onnxModel?.filename || null,
                fileUploadSetting.clusterTorchModel?.filename || null,
                fileUploadSetting.feature?.filename || null,
                fileUploadSetting.index?.filename || null,
                fileUploadSetting.isHalf

            )

            // サーバでロード中にキャッシュにセーブ
            try {
                const saveData: FileUploadSetting = {
                    pyTorchModel: fileUploadSetting.pyTorchModel ? { data: fileUploadSetting.pyTorchModel.data, filename: fileUploadSetting.pyTorchModel.filename } : null,
                    onnxModel: fileUploadSetting.onnxModel ? { data: fileUploadSetting.onnxModel.data, filename: fileUploadSetting.onnxModel.filename } : null,
                    configFile: fileUploadSetting.configFile ? { data: fileUploadSetting.configFile.data, filename: fileUploadSetting.configFile.filename } : null,
                    hubertTorchModel: fileUploadSetting.hubertTorchModel ? {
                        data: fileUploadSetting.hubertTorchModel.data, filename: fileUploadSetting.hubertTorchModel.filename
                    } : null,
                    clusterTorchModel: fileUploadSetting.clusterTorchModel ? {
                        data: fileUploadSetting.clusterTorchModel.data, filename: fileUploadSetting.clusterTorchModel.filename
                    } : null,
                    feature: fileUploadSetting.feature ? {
                        data: fileUploadSetting.feature.data, filename: fileUploadSetting.feature.filename
                    } : null,
                    index: fileUploadSetting.index ? {
                        data: fileUploadSetting.index.data, filename: fileUploadSetting.index.filename
                    } : null,
                    isHalf: fileUploadSetting.isHalf
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


    const getOnnx = async () => {
        return props.voiceChangerClient!.getOnnx()
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
        getOnnx,
    }
}