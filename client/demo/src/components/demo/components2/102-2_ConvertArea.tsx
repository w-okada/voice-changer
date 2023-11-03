import React, { useMemo } from "react";
import { useAppState } from "../../../001_provider/001_AppStateProvider";
import { useAppRoot } from "../../../001_provider/001_AppRootProvider";

export type ConvertProps = {
    inputChunkNums: number[];
};

export const ConvertArea = (props: ConvertProps) => {
    const { setting, serverSetting, setWorkletNodeSetting, trancateBuffer } = useAppState();
    const { appGuiSettingState } = useAppRoot();
    const edition = appGuiSettingState.edition;
    const convertArea = useMemo(() => {
        let nums: number[];
        if (!props.inputChunkNums) {
            nums = [8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 2048, 4096, 8192, 16384];
        } else {
            nums = props.inputChunkNums;
        }
        if (serverSetting.serverSetting.maxInputLength) {
            nums = nums.filter((x) => {
                return x < serverSetting.serverSetting.maxInputLength / 128;
            });
        }

        const gpusEntry = [...serverSetting.serverSetting.gpus];
        gpusEntry.push({
            id: -1,
            name: "cpu",
            memory: 0,
        });

        // const onClassName = serverSetting.serverSetting.gpu == 0 ? "config-sub-area-button-active" : "config-sub-area-button";
        // const offClassName = serverSetting.serverSetting.gpu == 0 ? "config-sub-area-button" : "config-sub-area-button-active";

        const cpuClassName = serverSetting.serverSetting.gpu == -1 ? "config-sub-area-button-active" : "config-sub-area-button";
        const gpu0ClassName = serverSetting.serverSetting.gpu == 0 ? "config-sub-area-button-active" : "config-sub-area-button";
        const gpu1ClassName = serverSetting.serverSetting.gpu == 1 ? "config-sub-area-button-active" : "config-sub-area-button";
        const gpu2ClassName = serverSetting.serverSetting.gpu == 2 ? "config-sub-area-button-active" : "config-sub-area-button";
        const gpu3ClassName = serverSetting.serverSetting.gpu == 3 ? "config-sub-area-button-active" : "config-sub-area-button";

        const gpuSelect =
            edition.indexOf("onnxdirectML-cuda") >= 0 ? (
                <>
                    <div className="config-sub-area-control">
                        <div className="config-sub-area-control-title">GPU(dml):</div>
                        <div className="config-sub-area-control-field">
                            <div className="config-sub-area-buttons">
                                <div
                                    onClick={async () => {
                                        await serverSetting.updateServerSettings({
                                            ...serverSetting.serverSetting,
                                            gpu: -1,
                                        });
                                    }}
                                    className={cpuClassName}
                                >
                                    <span className="config-sub-area-button-text-small">cpu</span>
                                </div>
                                <div
                                    onClick={async () => {
                                        await serverSetting.updateServerSettings({
                                            ...serverSetting.serverSetting,
                                            gpu: 0,
                                        });
                                    }}
                                    className={gpu0ClassName}
                                >
                                    <span className="config-sub-area-button-text-small">gpu0</span>
                                </div>
                                <div
                                    onClick={async () => {
                                        await serverSetting.updateServerSettings({
                                            ...serverSetting.serverSetting,
                                            gpu: 1,
                                        });
                                    }}
                                    className={gpu1ClassName}
                                >
                                    <span className="config-sub-area-button-text-small">gpu1</span>
                                </div>
                                <div
                                    onClick={async () => {
                                        await serverSetting.updateServerSettings({
                                            ...serverSetting.serverSetting,
                                            gpu: 2,
                                        });
                                    }}
                                    className={gpu2ClassName}
                                >
                                    <span className="config-sub-area-button-text-small">gpu2</span>
                                </div>
                                <div
                                    onClick={async () => {
                                        await serverSetting.updateServerSettings({
                                            ...serverSetting.serverSetting,
                                            gpu: 3,
                                        });
                                    }}
                                    className={gpu3ClassName}
                                >
                                    <span className="config-sub-area-button-text-small">gpu3</span>
                                </div>
                                <div className="config-sub-area-control">
                                    <span className="config-sub-area-button-text-small">
                                        <a href="https://github.com/w-okada/voice-changer/issues/410">more info</a>
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                </>
            ) : (
                <div className="config-sub-area-control">
                    <div className="config-sub-area-control-title">GPU:</div>
                    <div className="config-sub-area-control-field">
                        <select
                            className="body-select"
                            value={serverSetting.serverSetting.gpu}
                            onChange={(e) => {
                                serverSetting.updateServerSettings({ ...serverSetting.serverSetting, gpu: Number(e.target.value) });
                            }}
                        >
                            {gpusEntry.map((x) => {
                                return (
                                    <option key={x.id} value={x.id}>
                                        {x.name}
                                        {x.name == "cpu" ? "" : `(${(x.memory / 1024 / 1024 / 1024).toFixed(0)}GB)`}{" "}
                                    </option>
                                );
                            })}
                        </select>
                    </div>
                </div>
            );
        return (
            <div className="config-sub-area">
                <div className="config-sub-area-control">
                    <div className="config-sub-area-control-title">CHUNK:</div>
                    <div className="config-sub-area-control-field">
                        <select
                            className="body-select"
                            value={setting.workletNodeSetting.inputChunkNum}
                            onChange={(e) => {
                                setWorkletNodeSetting({ ...setting.workletNodeSetting, inputChunkNum: Number(e.target.value) });
                                trancateBuffer();
                                serverSetting.updateServerSettings({ ...serverSetting.serverSetting, serverReadChunkSize: Number(e.target.value) });
                            }}
                        >
                            {nums.map((x) => {
                                return (
                                    <option key={x} value={x}>
                                        {x} ({((x * 128 * 1000) / 48000).toFixed(1)} ms, {x * 128})
                                    </option>
                                );
                            })}
                        </select>
                    </div>
                </div>
                <div className="config-sub-area-control">
                    <div className="config-sub-area-control-title">EXTRA:</div>
                    <div className="config-sub-area-control-field">
                        <select
                            className="body-select"
                            value={serverSetting.serverSetting.extraConvertSize}
                            onChange={(e) => {
                                serverSetting.updateServerSettings({ ...serverSetting.serverSetting, extraConvertSize: Number(e.target.value) });
                                trancateBuffer();
                            }}
                        >
                            {[1024 * 4, 1024 * 8, 1024 * 16, 1024 * 32, 1024 * 64, 1024 * 128].map((x) => {
                                return (
                                    <option key={x} value={x}>
                                        {x}
                                    </option>
                                );
                            })}
                        </select>
                    </div>
                </div>
                {gpuSelect}
            </div>
        );
    }, [serverSetting.serverSetting, setting, serverSetting.updateServerSettings, setWorkletNodeSetting, edition]);

    return convertArea;
};
