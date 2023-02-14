import { useState, useMemo, useRef, useEffect } from "react"
import { VoiceChangerServerSetting, ServerInfo, Framework, OnnxExecutionProvider, DefaultVoiceChangerServerSetting, ServerSettingKey, INDEXEDDB_KEY_SERVER, INDEXEDDB_KEY_MODEL_DATA } from "../const"
import { VoiceChangerClient } from "../VoiceChangerClient"
import { useIndexedDB } from "./useIndexedDB"


// export type FileUploadSetting = {
//     pyTorchModel: File | null
//     configFile: File | null
//     onnxModel: File | null
// }

type ModelData = {
    file?: File
    data?: ArrayBuffer
    filename?: string
}

export type FileUploadSetting = {
    pyTorchModel: ModelData | null
    onnxModel: ModelData | null
    configFile: ModelData | null
}


const InitialFileUploadSetting: FileUploadSetting = {
    pyTorchModel: null,
    configFile: null,
    onnxModel: null,
}

export type UseServerSettingProps = {
    voiceChangerClient: VoiceChangerClient | null
}

export type ServerSettingState = {
    setting: VoiceChangerServerSetting;
    clearSetting: () => Promise<void>
    serverInfo: ServerInfo | undefined;
    fileUploadSetting: FileUploadSetting
    setFramework: (framework: Framework) => Promise<boolean>;
    setOnnxExecutionProvider: (provider: OnnxExecutionProvider) => Promise<boolean>;
    setSrcId: (num: number) => Promise<boolean>;
    setDstId: (num: number) => Promise<boolean>;
    setConvertChunkNum: (num: number) => Promise<boolean>;
    setMinConvertSize: (num: number) => Promise<boolean>
    setGpu: (num: number) => Promise<boolean>;
    setCrossFadeOffsetRate: (num: number) => Promise<boolean>;
    setCrossFadeEndRate: (num: number) => Promise<boolean>;
    setCrossFadeOverlapRate: (num: number) => Promise<boolean>;
    setF0Factor: (num: number) => Promise<boolean>;
    setF0Detector: (val: string) => Promise<boolean>;
    setRecordIO: (num: number) => Promise<boolean>;
    reloadServerInfo: () => Promise<void>;
    setFileUploadSetting: (val: FileUploadSetting) => void
    loadModel: () => Promise<void>
    uploadProgress: number
    isUploading: boolean
}

export const useServerSetting = (props: UseServerSettingProps): ServerSettingState => {
    const settingRef = useRef<VoiceChangerServerSetting>(DefaultVoiceChangerServerSetting)
    const [setting, _setSetting] = useState<VoiceChangerServerSetting>(settingRef.current)
    const [serverInfo, _setServerInfo] = useState<ServerInfo>()
    const [fileUploadSetting, setFileUploadSetting] = useState<FileUploadSetting>(InitialFileUploadSetting)
    const { setItem, getItem, removeItem } = useIndexedDB()


    // 初期化 その１ DBから取得
    useEffect(() => {
        const loadCache = async () => {
            const setting = await getItem(INDEXEDDB_KEY_SERVER)
            if (!setting) {
            } else {
                settingRef.current = setting as VoiceChangerServerSetting
            }
            _setSetting({ ...settingRef.current })

            const fileuploadSetting = await getItem(INDEXEDDB_KEY_MODEL_DATA)
            if (!fileuploadSetting) {
            } else {
                setFileUploadSetting(fileuploadSetting as FileUploadSetting)
            }
        }

        loadCache()
    }, [])
    // 初期化 その２ クライアントに設定
    useEffect(() => {
        if (!props.voiceChangerClient) return
        props.voiceChangerClient.updateServerSettings(ServerSettingKey.framework, setting.framework)
        props.voiceChangerClient.updateServerSettings(ServerSettingKey.onnxExecutionProvider, setting.onnxExecutionProvider)
        props.voiceChangerClient.updateServerSettings(ServerSettingKey.srcId, "" + setting.srcId)
        props.voiceChangerClient.updateServerSettings(ServerSettingKey.dstId, "" + setting.dstId)
        props.voiceChangerClient.updateServerSettings(ServerSettingKey.convertChunkNum, "" + setting.convertChunkNum)
        props.voiceChangerClient.updateServerSettings(ServerSettingKey.minConvertSize, "" + setting.minConvertSize)
        props.voiceChangerClient.updateServerSettings(ServerSettingKey.gpu, "" + setting.gpu)
        props.voiceChangerClient.updateServerSettings(ServerSettingKey.crossFadeOffsetRate, "" + setting.crossFadeOffsetRate)
        props.voiceChangerClient.updateServerSettings(ServerSettingKey.crossFadeEndRate, "" + setting.crossFadeEndRate)
        props.voiceChangerClient.updateServerSettings(ServerSettingKey.crossFadeOverlapRate, "" + setting.crossFadeOverlapRate)
        props.voiceChangerClient.updateServerSettings(ServerSettingKey.f0Factor, "" + setting.f0Factor)
        props.voiceChangerClient.updateServerSettings(ServerSettingKey.f0Detector, "" + setting.f0Detector)
        props.voiceChangerClient.updateServerSettings(ServerSettingKey.recordIO, "" + setting.recordIO)


    }, [props.voiceChangerClient])

    //////////////
    // 設定
    /////////////
    //// サーバに設定後、反映された情報と照合して値が一致していることを確認。一致していない場合はalert
    const _set_and_store = async (key: ServerSettingKey, newVal: string) => {
        if (!props.voiceChangerClient) return false

        const res = await props.voiceChangerClient.updateServerSettings(key, "" + newVal)

        _setServerInfo(res)
        if (newVal == res[key]) {
            const newSetting: VoiceChangerServerSetting = {
                ...settingRef.current,
                convertChunkNum: res.convertChunkNum,
                minConvertSize: res.minConvertSize,
                srcId: res.srcId,
                dstId: res.dstId,
                gpu: res.gpu,
                crossFadeOffsetRate: res.crossFadeOffsetRate,
                crossFadeEndRate: res.crossFadeEndRate,
                crossFadeOverlapRate: res.crossFadeOverlapRate,
                framework: res.framework,
                onnxExecutionProvider: (!!res.onnxExecutionProvider && res.onnxExecutionProvider.length > 0) ? res.onnxExecutionProvider[0] as OnnxExecutionProvider : DefaultVoiceChangerServerSetting.onnxExecutionProvider,
                f0Factor: res.f0Factor,
                f0Detector: res.f0Detector,
                recordIO: res.recordIO

            }
            _setSetting(newSetting)
            setItem(INDEXEDDB_KEY_SERVER, newSetting)
            return true
        } else {
            alert(`[ServerSetting] 設定が反映されていません([key:${key}, new:${newVal}, res:${res[key]}])。モデルの切り替えの場合、処理が非同期で行われるため反映されていないように見える場合があります。サーバコントロールのリロードボタンを押すとGUIに反映されるます。`)
            return false
        }

    }

    const setFramework = useMemo(() => {
        return async (framework: Framework) => {
            return await _set_and_store(ServerSettingKey.framework, "" + framework)
        }
    }, [props.voiceChangerClient])

    const setOnnxExecutionProvider = useMemo(() => {
        return async (provider: OnnxExecutionProvider) => {
            return await _set_and_store(ServerSettingKey.onnxExecutionProvider, "" + provider)
        }
    }, [props.voiceChangerClient])

    const setSrcId = useMemo(() => {
        return async (num: number) => {
            return await _set_and_store(ServerSettingKey.srcId, "" + num)
        }
    }, [props.voiceChangerClient])

    const setDstId = useMemo(() => {
        return async (num: number) => {
            return await _set_and_store(ServerSettingKey.dstId, "" + num)
        }
    }, [props.voiceChangerClient])

    const setConvertChunkNum = useMemo(() => {
        return async (num: number) => {
            return await _set_and_store(ServerSettingKey.convertChunkNum, "" + num)
        }
    }, [props.voiceChangerClient])

    const setMinConvertSize = useMemo(() => {
        return async (num: number) => {
            return await _set_and_store(ServerSettingKey.minConvertSize, "" + num)
        }
    }, [props.voiceChangerClient])


    const setGpu = useMemo(() => {
        return async (num: number) => {
            return await _set_and_store(ServerSettingKey.gpu, "" + num)
        }
    }, [props.voiceChangerClient])

    const setCrossFadeOffsetRate = useMemo(() => {
        return async (num: number) => {
            return await _set_and_store(ServerSettingKey.crossFadeOffsetRate, "" + num)
        }
    }, [props.voiceChangerClient])
    const setCrossFadeEndRate = useMemo(() => {
        return async (num: number) => {
            return await _set_and_store(ServerSettingKey.crossFadeEndRate, "" + num)
        }
    }, [props.voiceChangerClient])
    const setCrossFadeOverlapRate = useMemo(() => {
        return async (num: number) => {
            return await _set_and_store(ServerSettingKey.crossFadeOverlapRate, "" + num)
        }
    }, [props.voiceChangerClient])

    const setF0Factor = useMemo(() => {
        return async (num: number) => {
            return await _set_and_store(ServerSettingKey.f0Factor, "" + num)
        }
    }, [props.voiceChangerClient])

    const setF0Detector = useMemo(() => {
        return async (val: string) => {
            return await _set_and_store(ServerSettingKey.f0Detector, "" + val)
        }
    }, [props.voiceChangerClient])
    const setRecordIO = useMemo(() => {
        return async (num: number) => {
            return await _set_and_store(ServerSettingKey.recordIO, "" + num)
        }
    }, [props.voiceChangerClient])
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

            // ファイルをサーバにアップロード
            const models = [fileUploadSetting.onnxModel, fileUploadSetting.pyTorchModel].filter(x => { return x != null }) as ModelData[]
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

            const loadPromise = props.voiceChangerClient.loadModel(fileUploadSetting.configFile.filename!, fileUploadSetting.pyTorchModel?.filename || null, fileUploadSetting.onnxModel?.filename || null)

            // サーバでロード中にキャッシュにセーブ
            const saveData: FileUploadSetting = {
                pyTorchModel: fileUploadSetting.pyTorchModel ? { data: fileUploadSetting.pyTorchModel.data, filename: fileUploadSetting.pyTorchModel.filename } : null,
                onnxModel: fileUploadSetting.onnxModel ? { data: fileUploadSetting.onnxModel.data, filename: fileUploadSetting.onnxModel.filename } : null,
                configFile: { data: fileUploadSetting.configFile.data, filename: fileUploadSetting.configFile.filename }
            }
            setItem(INDEXEDDB_KEY_MODEL_DATA, saveData)

            await loadPromise
            setUploadProgress(0)
            setIsUploading(false)
            reloadServerInfo()
        }
    }, [fileUploadSetting, props.voiceChangerClient])

    // const _uploadFile = useMemo(() => {
    //     return async (file: File, onprogress: (progress: number, end: boolean) => void) => {
    //         if (!props.voiceChangerClient) return
    //         const num = await props.voiceChangerClient.uploadFile(file, onprogress)
    //         const res = await props.voiceChangerClient.concatUploadedFile(file, num)
    //         console.log("uploaded", num, res)
    //     }
    // }, [props.voiceChangerClient])
    // const loadModel = useMemo(() => {
    //     return async () => {
    //         if (!fileUploadSetting.pyTorchModel && !fileUploadSetting.onnxModel) {
    //             alert("PyTorchモデルとONNXモデルのどちらか一つ以上指定する必要があります。")
    //             return
    //         }
    //         if (!fileUploadSetting.configFile) {
    //             alert("Configファイルを指定する必要があります。")
    //             return
    //         }
    //         if (!props.voiceChangerClient) return


    //         setUploadProgress(0)
    //         setIsUploading(true)
    //         const models = [fileUploadSetting.pyTorchModel, fileUploadSetting.onnxModel].filter(x => { return x != null }) as File[]
    //         for (let i = 0; i < models.length; i++) {
    //             const progRate = 1 / models.length
    //             const progOffset = 100 * i * progRate
    //             await _uploadFile(models[i], (progress: number, _end: boolean) => {
    //                 // console.log(progress * progRate + progOffset, end, progRate,)
    //                 setUploadProgress(progress * progRate + progOffset)
    //             })
    //         }

    //         await _uploadFile(fileUploadSetting.configFile, (progress: number, end: boolean) => {
    //             console.log(progress, end)
    //         })

    //         await props.voiceChangerClient.loadModel(fileUploadSetting.configFile, fileUploadSetting.pyTorchModel, fileUploadSetting.onnxModel)
    //         setUploadProgress(0)
    //         setIsUploading(false)
    //         reloadServerInfo()
    //     }
    // }, [fileUploadSetting, props.voiceChangerClient])



    const reloadServerInfo = useMemo(() => {
        return async () => {
            if (!props.voiceChangerClient) return
            const res = await props.voiceChangerClient.getServerSettings()
            _setServerInfo(res)
            _setSetting({
                ...settingRef.current,
                convertChunkNum: res.convertChunkNum,
                srcId: res.srcId,
                dstId: res.dstId,
                gpu: res.gpu,
                crossFadeOffsetRate: res.crossFadeOffsetRate,
                crossFadeEndRate: res.crossFadeEndRate,
                crossFadeOverlapRate: res.crossFadeOverlapRate,
                framework: res.framework,
                onnxExecutionProvider: (!!res.onnxExecutionProvider && res.onnxExecutionProvider.length > 0) ? res.onnxExecutionProvider[0] as OnnxExecutionProvider : DefaultVoiceChangerServerSetting.onnxExecutionProvider,
                f0Factor: res.f0Factor,
                f0Detector: res.f0Detector,
                recordIO: res.recordIO
            })
        }
    }, [props.voiceChangerClient])

    const clearSetting = async () => {
        await removeItem(INDEXEDDB_KEY_SERVER)
        await removeItem(INDEXEDDB_KEY_MODEL_DATA)
    }


    return {
        setting,
        clearSetting,
        serverInfo,
        fileUploadSetting,
        setFramework,
        setOnnxExecutionProvider,
        setSrcId,
        setDstId,
        setConvertChunkNum,
        setMinConvertSize,
        setGpu,
        setCrossFadeOffsetRate,
        setCrossFadeEndRate,
        setCrossFadeOverlapRate,
        setF0Factor,
        setF0Detector,
        setRecordIO,
        reloadServerInfo,
        setFileUploadSetting,
        loadModel,
        uploadProgress,
        isUploading,
    }
}