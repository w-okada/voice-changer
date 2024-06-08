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

        const gpusEntry = serverSetting.serverSetting.gpus;

        const gpuSelect = (
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
                                        {x.memory ? `(${(x.memory / 1024 / 1024 / 1024).toFixed(0)}GB)` : ""}
                                    </option>
                                );
                            })}
                        </select>
                    </div>
                </div>
            );

        const extraArea = (
            <div className="config-sub-area-control">
                <div className="config-sub-area-control-title">EXTRA:</div>
                <div className="config-sub-area-control-field">
                    <div className="config-sub-area-slider-control">
                        <input
                            className="config-sub-area-slider-control-slider"
                            type="range"
                            min={0.1}
                            max={5}
                            step={0.1}
                            value={serverSetting.serverSetting.extraConvertSize}
                            onChange={(e) => {
                                serverSetting.updateServerSettings({ ...serverSetting.serverSetting, extraConvertSize: Number(e.target.value) });
                            }} />
                        <span className="config-sub-area-slider-control-val">{serverSetting.serverSetting.extraConvertSize} s</span>
                    </div>
                </div>
            </div>
        );
        return (
            <div className="config-sub-area">
                <div className="config-sub-area-control">
                    <div className="config-sub-area-control-title">CHUNK:</div>
                    <div className="config-sub-area-control-field">
                        <div className="config-sub-area-slider-control">
                            <input
                                className="config-sub-area-slider-control-slider"
                                type="range"
                                min={1}
                                max={1024}
                                step={1}
                                value={serverSetting.serverSetting.serverReadChunkSize}
                                onChange={(e) => {
                                    setWorkletNodeSetting({ ...setting.workletNodeSetting, inputChunkNum: Number(e.target.value) });
                                    trancateBuffer();
                                    serverSetting.updateServerSettings({ ...serverSetting.serverSetting, serverReadChunkSize: Number(e.target.value) });
                                }} />
                            <span className="config-sub-area-slider-control-val">{((serverSetting.serverSetting.serverReadChunkSize * 128 * 1000) / 48000).toFixed(1)} ms</span>
                        </div>
                    </div>
                </div>
                {extraArea}
                {gpuSelect}
            </div>
        );
    }, [serverSetting.serverSetting, setting, serverSetting.updateServerSettings, setWorkletNodeSetting, edition]);

    return convertArea;
};
