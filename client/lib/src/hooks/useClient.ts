import { useEffect, useMemo, useRef, useState } from "react";
import { VoiceChangerClient } from "../VoiceChangerClient";
import { useClientSetting } from "./useClientSetting";
import { IndexedDBStateAndMethod, useIndexedDB } from "./useIndexedDB";
import { ServerSettingState, useServerSetting } from "./useServerSetting";
import { useWorkletNodeSetting } from "./useWorkletNodeSetting";
import { useWorkletSetting } from "./useWorkletSetting";
import { ClientSetting, DefaultClientSettng, VoiceChangerClientSetting, WorkletNodeSetting, WorkletSetting } from "../const";

export type UseClientProps = {
    audioContext: AudioContext | null;
};

export type ClientState = {
    initialized: boolean;
    setting: ClientSetting;
    // 各種設定I/Fへの参照
    setVoiceChangerClientSetting: (_voiceChangerClientSetting: VoiceChangerClientSetting) => void;
    setServerUrl: (url: string) => void;
    start: () => Promise<void>;
    stop: () => Promise<void>;
    reloadClientSetting: () => Promise<void>;

    setWorkletNodeSetting: (_workletNodeSetting: WorkletNodeSetting) => void;
    startOutputRecording: () => void;
    stopOutputRecording: () => Promise<Float32Array>;
    trancateBuffer: () => Promise<void>;

    setWorkletSetting: (_workletSetting: WorkletSetting) => void;
    // workletSetting: WorkletSetting
    // workletSetting: WorkletSettingState
    // clientSetting: ClientSettingState
    // workletNodeSetting: WorkletNodeSettingState
    serverSetting: ServerSettingState;
    indexedDBState: IndexedDBStateAndMethod;

    // モニタリングデータ
    bufferingTime: number;
    volume: number;
    performance: PerformanceData;
    updatePerformance: (() => Promise<void>) | null;
    // setClientType: (val: ClientType) => void

    // 情報取得
    getInfo: () => Promise<void>;
    // 設定クリア
    clearSetting: () => Promise<void>;
    // AudioOutputElement  設定
    setAudioOutputElementId: (elemId: string) => void;
    setAudioMonitorElementId: (elemId: string) => void;

    ioErrorCount: number;
    resetIoErrorCount: () => void;
};

export type PerformanceData = {
    responseTime: number;
    preprocessTime: number;
    mainprocessTime: number;
    postprocessTime: number;
};
const InitialPerformanceData: PerformanceData = {
    responseTime: 0,
    preprocessTime: 0,
    mainprocessTime: 0,
    postprocessTime: 0,
};

export const useClient = (props: UseClientProps): ClientState => {
    const [initialized, setInitialized] = useState<boolean>(false);
    const [setting, setSetting] = useState<ClientSetting>(DefaultClientSettng);
    // (1-1) クライアント
    const voiceChangerClientRef = useRef<VoiceChangerClient | null>(null);
    const [voiceChangerClient, setVoiceChangerClient] = useState<VoiceChangerClient | null>(voiceChangerClientRef.current);
    //// クライアント初期化待ち用フラグ
    const initializedResolveRef = useRef<(value: void | PromiseLike<void>) => void>();
    const initializedPromise = useMemo(() => {
        return new Promise<void>((resolve) => {
            initializedResolveRef.current = resolve;
        });
    }, []);

    // (1-2) 各種設定I/F
    const voiceChangerClientSetting = useClientSetting({ voiceChangerClient, voiceChangerClientSetting: setting.voiceChangerClientSetting });
    const workletNodeSetting = useWorkletNodeSetting({ voiceChangerClient: voiceChangerClient, workletNodeSetting: setting.workletNodeSetting });
    useWorkletSetting({ voiceChangerClient, workletSetting: setting.workletSetting });
    const serverSetting = useServerSetting({ voiceChangerClient });
    const indexedDBState = useIndexedDB({ clientType: null });

    // (1-3) モニタリングデータ
    const [bufferingTime, setBufferingTime] = useState<number>(0);
    const [performance, setPerformance] = useState<PerformanceData>(InitialPerformanceData);
    const [volume, setVolume] = useState<number>(0);
    const [ioErrorCount, setIoErrorCount] = useState<number>(0);

    //// Server Audio Deviceを使うとき、モニタリングデータはpolling
    const updatePerformance = useMemo(() => {
        if (!voiceChangerClientRef.current) {
            return null;
        }

        return async () => {
            if (voiceChangerClientRef.current) {
                const performance = await voiceChangerClientRef.current!.getPerformance();
                const responseTime = performance[0];
                const preprocessTime = performance[1];
                const mainprocessTime = performance[2];
                const postprocessTime = performance[3];
                setPerformance({ responseTime, preprocessTime, mainprocessTime, postprocessTime });
            } else {
                const responseTime = 0;
                const preprocessTime = 0;
                const mainprocessTime = 0;
                const postprocessTime = 0;
                setPerformance({ responseTime, preprocessTime, mainprocessTime, postprocessTime });
            }
        };
    }, [voiceChangerClientRef.current]);

    // (1-4) エラーステータス
    const ioErrorCountRef = useRef<number>(0);
    const resetIoErrorCount = () => {
        ioErrorCountRef.current = 0;
        setIoErrorCount(ioErrorCountRef.current);
    };

    // 設定データ管理
    const { setItem, getItem, removeItem } = useIndexedDB({ clientType: null });
    // 設定データの更新と保存
    const _setSetting = (_setting: ClientSetting) => {
        const storeData = { ..._setting };
        storeData.voiceChangerClientSetting = { ...storeData.voiceChangerClientSetting };
        if (typeof storeData.voiceChangerClientSetting.audioInput != "string") {
            storeData.voiceChangerClientSetting.audioInput = "none";
        }
        setItem("clientSetting", storeData);

        setSetting(_setting);
    };
    // 設定データ初期化
    useEffect(() => {
        if (!voiceChangerClient) {
            return;
        }
        const loadCache = async () => {
            const _setting = (await getItem("clientSetting")) as ClientSetting;
            if (_setting) {
                setSetting(_setting);
                serverSetting.reloadServerInfo();
            }
        };
        loadCache();
    }, [voiceChangerClient]);

    // (2-1) クライアント初期化処理
    useEffect(() => {
        const initialized = async () => {
            if (!props.audioContext) {
                return;
            }
            const voiceChangerClient = new VoiceChangerClient(props.audioContext, true, {
                notifySendBufferingTime: (val: number) => {
                    setBufferingTime(val);
                },
                notifyResponseTime: (val: number, perf?: number[]) => {
                    const responseTime = val;
                    const preprocessTime = perf ? Math.ceil(perf[0] * 1000) : 0;
                    const mainprocessTime = perf ? Math.ceil(perf[1] * 1000) : 0;
                    const postprocessTime = perf ? Math.ceil(perf[2] * 1000) : 0;
                    setPerformance({ responseTime, preprocessTime, mainprocessTime, postprocessTime });
                },
                notifyException: (mes: string) => {
                    if (mes.length > 0) {
                        console.log(`error:${mes}`);
                        ioErrorCountRef.current += 1;
                        setIoErrorCount(ioErrorCountRef.current);
                    }
                },
                notifyVolume: (vol: number) => {
                    setVolume(vol);
                },
            });

            await voiceChangerClient.isInitialized();
            voiceChangerClientRef.current = voiceChangerClient;
            setVoiceChangerClient(voiceChangerClientRef.current);
            console.log("[useClient] client initialized");

            // const audio = document.getElementById(props.audioOutputElementId) as HTMLAudioElement
            // audio.srcObject = voiceChangerClientRef.current.stream
            // audio.play()
            initializedResolveRef.current!();
            setInitialized(true);
        };
        initialized();
    }, [props.audioContext]);

    const setAudioOutputElementId = (elemId: string) => {
        if (!voiceChangerClientRef.current) {
            console.warn("[voiceChangerClient] is not ready for set audio output.");
            return;
        }
        const audio = document.getElementById(elemId) as HTMLAudioElement;
        if (audio.paused) {
            audio.srcObject = voiceChangerClientRef.current.stream;
            audio.play();
        }
    };

    const setAudioMonitorElementId = (elemId: string) => {
        if (!voiceChangerClientRef.current) {
            console.warn("[voiceChangerClient] is not ready for set audio output.");
            return;
        }
        const audio = document.getElementById(elemId) as HTMLAudioElement;
        if (audio.paused) {
            audio.srcObject = voiceChangerClientRef.current.monitorStream;
            audio.play();
        }
    };

    // (2-2) 情報リロード
    const getInfo = useMemo(() => {
        return async () => {
            await initializedPromise;
            // FIXME: Hacky way to bring client chunk size in sync with server.
            await voiceChangerClientSetting.reloadClientSetting(); // 実質的な処理の意味はない
            const server = await serverSetting.reloadServerInfo();
            setWorkletNodeSetting({ ...setting.workletNodeSetting, inputChunkNum: server.serverReadChunkSize });
        };
    }, [voiceChangerClientSetting.reloadClientSetting, serverSetting.reloadServerInfo]);

    const clearSetting = async () => {
        await removeItem("clientSetting");
    };

    // 設定変更
    const setVoiceChangerClientSetting = (_voiceChangerClientSetting: VoiceChangerClientSetting) => {
        setting.voiceChangerClientSetting = _voiceChangerClientSetting;
        console.log("setting.voiceChangerClientSetting", setting.voiceChangerClientSetting);
        // workletSettingIF.setSetting(_workletSetting)
        _setSetting({ ...setting });
    };

    const setWorkletNodeSetting = (_workletNodeSetting: WorkletNodeSetting) => {
        setting.workletNodeSetting = _workletNodeSetting;
        console.log("setting.workletNodeSetting", setting.workletNodeSetting);
        // workletSettingIF.setSetting(_workletSetting)
        _setSetting({ ...setting });
    };

    const setWorkletSetting = (_workletSetting: WorkletSetting) => {
        setting.workletSetting = _workletSetting;
        console.log("setting.workletSetting", setting.workletSetting);
        // workletSettingIF.setSetting(_workletSetting)
        _setSetting({ ...setting });
    };

    return {
        initialized,
        setting,
        // 各種設定I/Fへの参照
        setVoiceChangerClientSetting,
        setServerUrl: voiceChangerClientSetting.setServerUrl,
        start: voiceChangerClientSetting.start,
        stop: voiceChangerClientSetting.stop,
        reloadClientSetting: voiceChangerClientSetting.reloadClientSetting,

        setWorkletNodeSetting,
        startOutputRecording: workletNodeSetting.startOutputRecording,
        stopOutputRecording: workletNodeSetting.stopOutputRecording,
        trancateBuffer: workletNodeSetting.trancateBuffer,

        setWorkletSetting,
        // workletSetting: workletSettingIF.setting,
        serverSetting,
        indexedDBState,

        // モニタリングデータ
        bufferingTime,
        volume,
        performance,
        updatePerformance,

        // 情報取得
        getInfo,

        // 設定クリア
        clearSetting,

        // AudioOutputElement  設定
        setAudioOutputElementId,
        setAudioMonitorElementId,

        ioErrorCount,
        resetIoErrorCount,
    };
};
