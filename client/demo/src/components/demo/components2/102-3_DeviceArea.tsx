import React, { useEffect, useMemo, useRef, useState } from "react";
import { useAppState } from "../../../001_provider/001_AppStateProvider";
import { fileSelectorAsDataURL, useIndexedDB } from "@dannadori/voice-changer-client-js";
import { useGuiState } from "../001_GuiStateProvider";
import { AUDIO_ELEMENT_FOR_PLAY_MONITOR, AUDIO_ELEMENT_FOR_PLAY_RESULT, AUDIO_ELEMENT_FOR_TEST_CONVERTED, AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK, AUDIO_ELEMENT_FOR_TEST_ORIGINAL, INDEXEDDB_KEY_AUDIO_MONITR, INDEXEDDB_KEY_AUDIO_OUTPUT } from "../../../const";
import { isDesktopApp } from "../../../const";

export type DeviceAreaProps = {};

export const DeviceArea = (_props: DeviceAreaProps) => {
    const { setting, serverSetting, audioContext, setAudioOutputElementId, setAudioMonitorElementId, initializedRef, setVoiceChangerClientSetting, startOutputRecording, stopOutputRecording } = useAppState();
    const { isConverting, audioInputForGUI, inputAudioDeviceInfo, setAudioInputForGUI, fileInputEchoback, setFileInputEchoback, setAudioOutputForGUI, setAudioMonitorForGUI, audioOutputForGUI, audioMonitorForGUI, outputAudioDeviceInfo, shareScreenEnabled, setShareScreenEnabled } = useGuiState();
    const [inputHostApi, setInputHostApi] = useState<string>("ALL");
    const [outputHostApi, setOutputHostApi] = useState<string>("ALL");
    const [monitorHostApi, setMonitorHostApi] = useState<string>("ALL");
    const audioSrcNode = useRef<MediaElementAudioSourceNode>();
    const displayMediaStream = useRef<MediaStream | null>(null);

    const { getItem, setItem } = useIndexedDB({ clientType: null });
    const [outputRecordingStarted, setOutputRecordingStarted] = useState<boolean>(false);

    // (1) Audio Mode
    const deviceModeRow = useMemo(() => {
        const enableServerAudio = serverSetting.serverSetting.enableServerAudio;
        const clientChecked = enableServerAudio == 1 ? false : true;
        const serverChecked = enableServerAudio == 1 ? true : false;

        const onDeviceModeChanged = (val: number) => {
            if (isConverting) {
                alert("cannot change mode when voice conversion is enabled");
                return;
            }
            serverSetting.updateServerSettings({ ...serverSetting.serverSetting, enableServerAudio: val });
        };

        return (
            <div className="config-sub-area-control">
                <div className="config-sub-area-control-title">AUDIO:</div>
                <div className="config-sub-area-control-field">
                    <div className="config-sub-area-noise-container">
                        <div className="config-sub-area-noise-checkbox-container">
                            <input
                                type="radio"
                                id="client-device"
                                name="device-mode"
                                checked={clientChecked}
                                onChange={() => {
                                    onDeviceModeChanged(0);
                                }}
                            />{" "}
                            <label htmlFor="client-device">client</label>
                        </div>
                        <div className="config-sub-area-noise-checkbox-container">
                            <input
                                className="left-padding-1"
                                type="radio"
                                id="server-device"
                                name="device-mode"
                                checked={serverChecked}
                                onChange={() => {
                                    onDeviceModeChanged(1);
                                }}
                            />
                            <label htmlFor="server-device">server</label>
                        </div>
                    </div>
                </div>
            </div>
        );
    }, [serverSetting.serverSetting, serverSetting.updateServerSettings, isConverting]);

    // (2) Audio Input
    // キャッシュの設定は反映（たぶん、設定操作の時も起動していしまう。が問題は起こらないはず）
    useEffect(() => {
        if (typeof setting.voiceChangerClientSetting.audioInput == "string") {
            if (
                inputAudioDeviceInfo.find((x) => {
                    // console.log("COMPARE:", x.deviceId, appState.clientSetting.setting.audioInput)
                    return x.deviceId == setting.voiceChangerClientSetting.audioInput;
                })
            ) {
                setAudioInputForGUI(setting.voiceChangerClientSetting.audioInput);
            }
        }
    }, [inputAudioDeviceInfo, setting.voiceChangerClientSetting.audioInput]);

    // (2-1) クライアント
    const clientAudioInputRow = useMemo(() => {
        if (serverSetting.serverSetting.enableServerAudio == 1) {
            return <></>;
        }

        return (
            <div className="config-sub-area-control">
                <div className="config-sub-area-control-title left-padding-1">input</div>
                <div className="config-sub-area-control-field">
                    <select
                        className="body-select"
                        value={audioInputForGUI}
                        onChange={async (e) => {
                            setAudioInputForGUI(e.target.value);
                            if (e.target.value != "file" && e.target.value != "screen") {
                                try {
                                    await setVoiceChangerClientSetting({ ...setting.voiceChangerClientSetting, audioInput: e.target.value });
                                } catch (e) {
                                    alert(e);
                                    console.error(e);
                                    setAudioInputForGUI("none");
                                    await setVoiceChangerClientSetting({ ...setting.voiceChangerClientSetting, audioInput: null });
                                }
                            }
                        }}
                    >
                        {inputAudioDeviceInfo.map((x) => {
                            return (
                                <option key={x.deviceId} value={x.deviceId}>
                                    {x.label}
                                </option>
                            );
                        })}
                    </select>
                </div>
            </div>
        );
    }, [setVoiceChangerClientSetting, setting, inputAudioDeviceInfo, audioInputForGUI, serverSetting.serverSetting.enableServerAudio]);

    // (2-2) サーバ
    const serverAudioInputRow = useMemo(() => {
        if (serverSetting.serverSetting.enableServerAudio == 0) {
            return <></>;
        }

        const devices = serverSetting.serverSetting.serverAudioInputDevices;
        const hostAPIs = new Set(
            devices.map((x) => {
                return x.hostAPI;
            }),
        );
        const hostAPIOptions = Array.from(hostAPIs).map((x, index) => {
            return (
                <option value={x} key={index}>
                    {x}
                </option>
            );
        });

        const filteredDevice = devices
            .map((x, index) => {
                if (inputHostApi != "ALL" && x.hostAPI != inputHostApi) {
                    return null;
                }
                return (
                    <option value={x.index} key={index}>
                        [{x.hostAPI}]{x.name}
                    </option>
                );
            })
            .filter((x) => x != null);
        const currentValue = devices.find((x) => {
            return (x.hostAPI == inputHostApi || inputHostApi == "ALL") && x.index == serverSetting.serverSetting.serverInputDeviceId;
        })
            ? serverSetting.serverSetting.serverInputDeviceId
            : -1;

        return (
            <div className="config-sub-area-control">
                <div className="config-sub-area-control-title left-padding-1">input</div>
                <div className="config-sub-area-control-field">
                    <div className="config-sub-area-control-field-auido-io">
                        <select
                            className="config-sub-area-control-field-auido-io-filter"
                            name="kinds"
                            id="kinds"
                            value={inputHostApi}
                            onChange={(e) => {
                                setInputHostApi(e.target.value);
                            }}
                        >
                            <option value="ALL" key="ALL">
                                ALL
                            </option>
                            {hostAPIOptions}
                        </select>
                        <select
                            className="config-sub-area-control-field-auido-io-select"
                            value={currentValue}
                            onChange={(e) => {
                                serverSetting.updateServerSettings({ ...serverSetting.serverSetting, serverInputDeviceId: Number(e.target.value) });
                            }}
                        >
                            {filteredDevice}
                            <option value="-1" key="none">
                                none
                            </option>
                        </select>
                    </div>
                </div>
            </div>
        );
    }, [inputHostApi, serverSetting.serverSetting, serverSetting.updateServerSettings, serverSetting.serverSetting.enableServerAudio]);

    // (2-3) File
    useEffect(() => {
        [AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK].forEach((x) => {
            const audio = document.getElementById(x) as HTMLAudioElement;
            if (audio) {
                audio.volume = fileInputEchoback ? 1 : 0;
            }
        });
    }, [fileInputEchoback]);

    const audioInputMediaRow = useMemo(() => {
        if (audioInputForGUI != "file" || serverSetting.serverSetting.enableServerAudio == 1) {
            return <></>;
        }

        const onFileLoadClicked = async () => {
            const url = await fileSelectorAsDataURL("");

            // input stream for client.
            const audio = document.getElementById(AUDIO_ELEMENT_FOR_TEST_CONVERTED) as HTMLAudioElement;
            audio.pause();
            audio.srcObject = null;
            audio.src = url;
            await audio.play();
            if (!audioSrcNode.current) {
                audioSrcNode.current = audioContext!.createMediaElementSource(audio);
            }
            if (audioSrcNode.current.mediaElement != audio) {
                audioSrcNode.current = audioContext!.createMediaElementSource(audio);
            }

            const dst = audioContext.createMediaStreamDestination();
            audioSrcNode.current.connect(dst);
            try {
                setVoiceChangerClientSetting({ ...setting.voiceChangerClientSetting, audioInput: dst.stream });
            } catch (e) {
                console.error(e);
            }

            const audio_echo = document.getElementById(AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK) as HTMLAudioElement;
            audio_echo.srcObject = dst.stream;
            audio_echo.play();
            audio_echo.volume = 0;
            setFileInputEchoback(false);

            // // original stream to play.
            // const audio_org = document.getElementById(AUDIO_ELEMENT_FOR_TEST_ORIGINAL) as HTMLAudioElement;
            // audio_org.src = url;
            // audio_org.pause();
        };

        const echobackClass = fileInputEchoback ? "config-sub-area-control-field-wav-file-echoback-button-active" : "config-sub-area-control-field-wav-file-echoback-button";
        return (
            <div className="config-sub-area-control">
                <div className="config-sub-area-control-field">
                    <div className="config-sub-area-control-field-wav-file left-padding-1">
                        <div className="config-sub-area-control-field-wav-file-audio-container">
                            {/* <audio id={AUDIO_ELEMENT_FOR_TEST_ORIGINAL} controls hidden></audio> */}
                            <audio className="config-sub-area-control-field-wav-file-audio" id={AUDIO_ELEMENT_FOR_TEST_CONVERTED} controls controlsList="nodownload noplaybackrate"></audio>
                            <audio id={AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK} controls hidden></audio>
                        </div>
                        <div>
                            <img className="config-sub-area-control-field-wav-file-folder" src="./assets/icons/folder.svg" onClick={onFileLoadClicked} />
                        </div>
                        <div
                            className={echobackClass}
                            onClick={() => {
                                setFileInputEchoback(!fileInputEchoback);
                            }}
                        >
                            echo{fileInputEchoback}
                        </div>
                    </div>
                </div>
            </div>
        );
    }, [audioInputForGUI, fileInputEchoback, setting, serverSetting.serverSetting]);

    const audioInputScreenRow = useMemo(() => {
        if (audioInputForGUI != "screen" || serverSetting.serverSetting.enableServerAudio == 1) {
            return <></>;
        }

        const onSelectScreenClicked = async () => {
            // 既存msをクローズ
            if (displayMediaStream.current) {
                displayMediaStream.current.getTracks().forEach((x) => {
                    x.stop();
                });
                displayMediaStream.current = null;
            }

            // 共有ストップ
            if (shareScreenEnabled == true) {
                setShareScreenEnabled(false);
                return;
            }

            // 共有スタート
            try {
                if (isDesktopApp()) {
                    const constraints = {
                        audio: {
                            mandatory: {
                                chromeMediaSource: "desktop",
                            },
                        },
                        video: {
                            mandatory: {
                                chromeMediaSource: "desktop",
                            },
                        },
                    };
                    // @ts-ignore
                    displayMediaStream.current = await navigator.mediaDevices.getUserMedia(constraints);
                } else {
                    displayMediaStream.current = await navigator.mediaDevices.getDisplayMedia({
                        video: true,
                        audio: true,
                    });
                }
            } catch (e) {
                console.log(e);
                return;
            }
            if (!displayMediaStream.current) {
                console.log("no ms");
                return;
            }
            if (displayMediaStream.current?.getAudioTracks().length == 0) {
                displayMediaStream.current.getTracks().forEach((x) => {
                    x.stop();
                });
                displayMediaStream.current = null;
                console.log("no audio track");
                return;
            }

            try {
                setVoiceChangerClientSetting({ ...setting.voiceChangerClientSetting, audioInput: displayMediaStream.current });
            } catch (e) {
                console.error(e);
            }

            setShareScreenEnabled(!shareScreenEnabled);
        };

        const echobackClass = shareScreenEnabled ? "config-sub-area-control-field-screen-select-button-active" : "config-sub-area-control-field-screen-select-button";
        return (
            <div className="config-sub-area-control">
                <div className="config-sub-area-control-title left-padding-1"></div>
                <div className="config-sub-area-control-field">
                    <div className="config-sub-area-control-field-screen-select">
                        <div
                            className={echobackClass}
                            onClick={() => {
                                onSelectScreenClicked();
                            }}
                        >
                            capture
                        </div>
                    </div>
                </div>
            </div>
        );
    }, [audioInputForGUI, setting, serverSetting.serverSetting, shareScreenEnabled, setShareScreenEnabled]);

    // (3) Audio Output
    useEffect(() => {
        const loadCache = async () => {
            const key = await getItem(INDEXEDDB_KEY_AUDIO_OUTPUT);
            if (key) {
                setAudioOutputForGUI(key as string);
            }
        };
        loadCache();
    }, []);

    useEffect(() => {
        const setAudioOutput = async () => {
            const mediaDeviceInfos = await navigator.mediaDevices.enumerateDevices();

            // [AUDIO_ELEMENT_FOR_PLAY_RESULT, AUDIO_ELEMENT_FOR_TEST_ORIGINAL, AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK].forEach((x) => {
            [AUDIO_ELEMENT_FOR_PLAY_RESULT, AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK].forEach((x) => {
                const audio = document.getElementById(x) as HTMLAudioElement;
                if (audio) {
                    if (serverSetting.serverSetting.enableServerAudio == 1) {
                        // Server Audio を使う場合はElementから音は出さない。
                        audio.volume = 0;
                    } else if (audioOutputForGUI == "none") {
                        try {
                            // @ts-ignore
                            audio.setSinkId("");
                        } catch (e) {
                            console.error("catch:" + e);
                        }
                        if (x == AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK) {
                            audio.volume = 0;
                        } else {
                            audio.volume = 0;
                        }
                    } else {
                        const audioOutputs = mediaDeviceInfos.filter((x) => {
                            return x.kind == "audiooutput";
                        });
                        const found = audioOutputs.some((x) => {
                            return x.deviceId == audioOutputForGUI;
                        });
                        if (found) {
                            try {
                                // @ts-ignore // 例外キャッチできないので事前にIDチェックが必要らしい。！？
                                audio.setSinkId(audioOutputForGUI);
                            } catch (e) {
                                console.error("catch:" + e);
                            }
                        } else {
                            console.warn("No audio output device. use default");
                        }

                        if (x == AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK) {
                            audio.volume = fileInputEchoback ? 1 : 0;
                        } else {
                            audio.volume = 1;
                        }
                    }
                }
            });
        };
        setAudioOutput();
    }, [audioOutputForGUI, fileInputEchoback, serverSetting.serverSetting.enableServerAudio]);

    // (3-1) クライアント
    const clientAudioOutputRow = useMemo(() => {
        if (serverSetting.serverSetting.enableServerAudio == 1) {
            return <></>;
        }

        return (
            <div className="config-sub-area-control">
                <div className="config-sub-area-control-title left-padding-1">output</div>
                <div className="config-sub-area-control-field">
                    <select
                        className="body-select"
                        value={audioOutputForGUI}
                        onChange={(e) => {
                            setAudioOutputForGUI(e.target.value);
                            setItem(INDEXEDDB_KEY_AUDIO_OUTPUT, e.target.value);
                        }}
                    >
                        {outputAudioDeviceInfo.map((x) => {
                            return (
                                <option key={x.deviceId} value={x.deviceId}>
                                    {x.label}
                                </option>
                            );
                        })}
                    </select>
                </div>
            </div>
        );
    }, [serverSetting.serverSetting.enableServerAudio, outputAudioDeviceInfo, audioOutputForGUI]);

    useEffect(() => {
        console.log("initializedRef.current", initializedRef.current);
        setAudioOutputElementId(AUDIO_ELEMENT_FOR_PLAY_RESULT);
    }, [initializedRef.current]);

    // (3-2) サーバ
    const serverAudioOutputRow = useMemo(() => {
        if (serverSetting.serverSetting.enableServerAudio == 0) {
            return <></>;
        }
        const devices = serverSetting.serverSetting.serverAudioOutputDevices;
        const hostAPIs = new Set(
            devices.map((x) => {
                return x.hostAPI;
            }),
        );
        const hostAPIOptions = Array.from(hostAPIs).map((x, index) => {
            return (
                <option value={x} key={index}>
                    {x}
                </option>
            );
        });

        const filteredDevice = devices
            .map((x, index) => {
                if (outputHostApi != "ALL" && x.hostAPI != outputHostApi) {
                    return null;
                }
                return (
                    <option value={x.index} key={index}>
                        [{x.hostAPI}]{x.name}
                    </option>
                );
            })
            .filter((x) => x != null);

        const currentValue = devices.find((x) => {
            return (x.hostAPI == outputHostApi || outputHostApi == "ALL") && x.index == serverSetting.serverSetting.serverOutputDeviceId;
        })
            ? serverSetting.serverSetting.serverOutputDeviceId
            : -1;

        return (
            <div className="config-sub-area-control">
                <div className="config-sub-area-control-title left-padding-1">output</div>
                <div className="config-sub-area-control-field">
                    <div className="config-sub-area-control-field-auido-io">
                        <select
                            className="config-sub-area-control-field-auido-io-filter"
                            name="kinds"
                            id="kinds"
                            value={outputHostApi}
                            onChange={(e) => {
                                setOutputHostApi(e.target.value);
                            }}
                        >
                            <option value="ALL" key="ALL">
                                ALL
                            </option>
                            {hostAPIOptions}
                        </select>
                        <select
                            className="config-sub-area-control-field-auido-io-select"
                            value={currentValue}
                            onChange={(e) => {
                                serverSetting.updateServerSettings({ ...serverSetting.serverSetting, serverOutputDeviceId: Number(e.target.value) });
                            }}
                        >
                            {filteredDevice}
                            <option value="-1" key="none">
                                none
                            </option>
                        </select>
                    </div>
                </div>
            </div>
        );
    }, [outputHostApi, serverSetting.serverSetting, serverSetting.updateServerSettings, serverSetting.serverSetting.enableServerAudio]);

    // (4) レコーダー
    const outputRecorderRow = useMemo(() => {
        if (serverSetting.serverSetting.enableServerAudio == 1) {
            return <></>;
        }
        const onOutputRecordStartClicked = async () => {
            setOutputRecordingStarted(true);
            await startOutputRecording();
        };
        const onOutputRecordStopClicked = async () => {
            setOutputRecordingStarted(false);
            const record = await stopOutputRecording();
            downloadRecord(record);
        };

        const startClassName = outputRecordingStarted ? "config-sub-area-button-active" : "config-sub-area-button";
        const stopClassName = outputRecordingStarted ? "config-sub-area-button" : "config-sub-area-button-active";
        return (
            <div className="config-sub-area-control">
                <div className="config-sub-area-control-title">REC.</div>
                <div className="config-sub-area-control-field">
                    <div className="config-sub-area-buttons">
                        <div onClick={onOutputRecordStartClicked} className={startClassName}>
                            start
                        </div>
                        <div onClick={onOutputRecordStopClicked} className={stopClassName}>
                            stop
                        </div>
                    </div>
                </div>
            </div>
        );
    }, [outputRecordingStarted, startOutputRecording, stopOutputRecording, serverSetting.serverSetting.enableServerAudio]);

    // (5) サンプリングレート
    const sampleRateRow = useMemo(() => {
        if (serverSetting.serverSetting.enableServerAudio == 0) {
            return <></>;
        }

        return (
            <div className="config-sub-area-control">
                <div className="config-sub-area-control-title left-padding-1">S.R.</div>
                <div className="config-sub-area-control-field">
                    <div className="config-sub-area-control-field-auido-io">
                        <select
                            className="config-sub-area-control-field-sample-rate-select"
                            value={serverSetting.serverSetting.serverAudioSampleRate}
                            onChange={(e) => {
                                serverSetting.updateServerSettings({ ...serverSetting.serverSetting, serverAudioSampleRate: Number(e.target.value) });
                            }}
                        >
                            {[16000, 32000, 44100, 48000, 96000, 192000].map((x) => {
                                return (
                                    <option key={x} value={x}>
                                        {x}
                                    </option>
                                );
                            })}
                        </select>
                    </div>
                </div>
            </div>
        );
    }, [serverSetting.serverSetting, serverSetting.updateServerSettings, serverSetting.serverSetting.enableServerAudio]);

    // (6) モニター
    useEffect(() => {
        const loadCache = async () => {
            const key = await getItem(INDEXEDDB_KEY_AUDIO_MONITR);
            if (key) {
                setAudioMonitorForGUI(key as string);
            }
        };
        loadCache();
    }, []);
    useEffect(() => {
        const setAudioMonitor = async () => {
            const mediaDeviceInfos = await navigator.mediaDevices.enumerateDevices();

            [AUDIO_ELEMENT_FOR_PLAY_MONITOR].forEach((x) => {
                const audio = document.getElementById(x) as HTMLAudioElement;
                if (audio) {
                    if (serverSetting.serverSetting.enableServerAudio == 1) {
                        // Server Audio を使う場合はElementから音は出さない。
                        audio.volume = 0;
                    } else if (audioMonitorForGUI == "none") {
                        try {
                            // @ts-ignore
                            audio.setSinkId("");
                            audio.volume = 0;
                        } catch (e) {
                            console.error("catch:" + e);
                        }
                    } else {
                        const audioOutputs = mediaDeviceInfos.filter((x) => {
                            return x.kind == "audiooutput";
                        });
                        const found = audioOutputs.some((x) => {
                            return x.deviceId == audioMonitorForGUI;
                        });
                        if (found) {
                            try {
                                // @ts-ignore // 例外キャッチできないので事前にIDチェックが必要らしい。！？
                                audio.setSinkId(audioMonitorForGUI);
                                audio.volume = 1;
                            } catch (e) {
                                console.error("catch:" + e);
                            }
                        } else {
                            console.warn("No audio output device. use default");
                        }
                    }
                }
            });
        };
        setAudioMonitor();
    }, [audioMonitorForGUI, serverSetting.serverSetting.enableServerAudio]);

    // (6-1) クライアント
    const clientMonitorRow = useMemo(() => {
        if (serverSetting.serverSetting.enableServerAudio == 1) {
            return <></>;
        }

        return (
            <div className="config-sub-area-control">
                <div className="config-sub-area-control-title left-padding-1">monitor</div>
                <div className="config-sub-area-control-field">
                    <select
                        className="body-select"
                        value={audioMonitorForGUI}
                        onChange={(e) => {
                            setAudioMonitorForGUI(e.target.value);
                            setItem(INDEXEDDB_KEY_AUDIO_MONITR, e.target.value);
                        }}
                    >
                        {outputAudioDeviceInfo.map((x) => {
                            return (
                                <option key={x.deviceId} value={x.deviceId}>
                                    {x.label}
                                </option>
                            );
                        })}
                    </select>
                </div>
            </div>
        );
    }, [serverSetting.serverSetting.enableServerAudio, outputAudioDeviceInfo, audioMonitorForGUI]);

    useEffect(() => {
        console.log("initializedRef.current", initializedRef.current);
        setAudioMonitorElementId(AUDIO_ELEMENT_FOR_PLAY_MONITOR);
    }, [initializedRef.current]);

    // (6-2) サーバ
    const serverMonitorRow = useMemo(() => {
        if (serverSetting.serverSetting.enableServerAudio == 0) {
            return <></>;
        }
        const devices = serverSetting.serverSetting.serverAudioOutputDevices;
        const hostAPIs = new Set(
            devices.map((x) => {
                return x.hostAPI;
            }),
        );
        const hostAPIOptions = Array.from(hostAPIs).map((x, index) => {
            return (
                <option value={x} key={index}>
                    {x}
                </option>
            );
        });

        const filteredDevice = devices
            .map((x, index) => {
                if (monitorHostApi != "ALL" && x.hostAPI != monitorHostApi) {
                    return null;
                }
                return (
                    <option value={x.index} key={index}>
                        [{x.hostAPI}]{x.name}
                    </option>
                );
            })
            .filter((x) => x != null);
        filteredDevice.unshift(
            <option value={-1} key={-1}>
                none
            </option>,
        );

        const currentValue = devices.find((x) => {
            return (x.hostAPI == monitorHostApi || monitorHostApi == "ALL") && x.index == serverSetting.serverSetting.serverMonitorDeviceId;
        })
            ? serverSetting.serverSetting.serverMonitorDeviceId
            : -1;

        return (
            <div className="config-sub-area-control">
                <div className="config-sub-area-control-title left-padding-1">monitor</div>
                <div className="config-sub-area-control-field">
                    <div className="config-sub-area-control-field-auido-io">
                        <select
                            className="config-sub-area-control-field-auido-io-filter"
                            name="kinds"
                            id="kinds"
                            value={monitorHostApi}
                            onChange={(e) => {
                                setMonitorHostApi(e.target.value);
                            }}
                        >
                            <option value="ALL" key="ALL">
                                ALL
                            </option>
                            {hostAPIOptions}
                        </select>
                        <select
                            className="config-sub-area-control-field-auido-io-select"
                            value={currentValue}
                            onChange={(e) => {
                                serverSetting.updateServerSettings({ ...serverSetting.serverSetting, serverMonitorDeviceId: Number(e.target.value) });
                            }}
                        >
                            {filteredDevice}
                        </select>
                    </div>
                </div>
            </div>
        );
    }, [monitorHostApi, serverSetting.serverSetting, serverSetting.updateServerSettings, serverSetting.serverSetting.enableServerAudio]);

    const monitorGainControl = useMemo(() => {
        const currentMonitorGain = serverSetting.serverSetting.enableServerAudio == 0 ? setting.voiceChangerClientSetting.monitorGain : serverSetting.serverSetting.serverMonitorAudioGain;
        const monitorValueUpdatedAction =
            serverSetting.serverSetting.enableServerAudio == 0
                ? async (val: number) => {
                      await setVoiceChangerClientSetting({ ...setting.voiceChangerClientSetting, monitorGain: val });
                  }
                : async (val: number) => {
                      await serverSetting.updateServerSettings({ ...serverSetting.serverSetting, serverMonitorAudioGain: val });
                  };

        return (
            <div className="config-sub-area-control">
                <div className="config-sub-area-control-title left-padding-2">gain</div>
                <div className="config-sub-area-control-field">
                    <div className="config-sub-area-control-field-auido-io">
                        <span className="character-area-slider-control-slider">
                            <input
                                type="range"
                                min="0.1"
                                max="10.0"
                                step="0.1"
                                value={currentMonitorGain}
                                onChange={(e) => {
                                    monitorValueUpdatedAction(Number(e.target.value));
                                }}
                            ></input>
                        </span>
                        <span className="character-area-slider-control-val">{currentMonitorGain}</span>
                    </div>
                </div>
            </div>
        );
    }, [serverSetting.serverSetting, setting, setVoiceChangerClientSetting, serverSetting.updateServerSettings]);

    return (
        <div className="config-sub-area">
            {deviceModeRow}
            {sampleRateRow}
            {clientAudioInputRow}
            {serverAudioInputRow}
            {audioInputMediaRow}
            {audioInputScreenRow}
            {clientAudioOutputRow}
            {serverAudioOutputRow}
            {clientMonitorRow}
            {serverMonitorRow}
            {monitorGainControl}

            {outputRecorderRow}
            <audio hidden id={AUDIO_ELEMENT_FOR_PLAY_RESULT}></audio>
            <audio hidden id={AUDIO_ELEMENT_FOR_PLAY_MONITOR}></audio>
        </div>
    );
};

const downloadRecord = (data: Float32Array) => {
    const writeString = (view: DataView, offset: number, string: string) => {
        for (var i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    const floatTo16BitPCM = (output: DataView, offset: number, input: Float32Array) => {
        for (var i = 0; i < input.length; i++, offset += 2) {
            var s = Math.max(-1, Math.min(1, input[i]));
            output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        }
    };

    const buffer = new ArrayBuffer(44 + data.length * 2);
    const view = new DataView(buffer);

    // https://www.youfit.co.jp/archives/1418
    writeString(view, 0, "RIFF"); // RIFFヘッダ
    view.setUint32(4, 32 + data.length * 2, true); // これ以降のファイルサイズ
    writeString(view, 8, "WAVE"); // WAVEヘッダ
    writeString(view, 12, "fmt "); // fmtチャンク
    view.setUint32(16, 16, true); // fmtチャンクのバイト数
    view.setUint16(20, 1, true); // フォーマットID
    view.setUint16(22, 1, true); // チャンネル数
    view.setUint32(24, 48000, true); // サンプリングレート
    view.setUint32(28, 48000 * 2, true); // データ速度
    view.setUint16(32, 2, true); // ブロックサイズ
    view.setUint16(34, 16, true); // サンプルあたりのビット数
    writeString(view, 36, "data"); // dataチャンク
    view.setUint32(40, data.length * 2, true); // 波形データのバイト数
    floatTo16BitPCM(view, 44, data); // 波形データ
    const audioBlob = new Blob([view], { type: "audio/wav" });

    const url = URL.createObjectURL(audioBlob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `output.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
};
