import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useAppRoot } from "../../../001_provider/001_AppRootProvider"

export type ConvertProps = {
    inputChunkNums: number[]
}


export const ConvertArea = (props: ConvertProps) => {
    const { setting, serverSetting, setWorkletNodeSetting, trancateBuffer } = useAppState()
    const { appGuiSettingState } = useAppRoot()
    const edition = appGuiSettingState.edition
    const convertArea = useMemo(() => {
        let nums: number[]
        if (!props.inputChunkNums) {
            nums = [8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 2048]
        } else {
            nums = props.inputChunkNums
        }

        const gpusEntry = [...serverSetting.serverSetting.gpus]
        gpusEntry.push({
            id: -1,
            name: "cpu",
            memory: 0
        })
        const gpuSelect = edition.indexOf("onnxdirectML-cuda") >= 0 ? (
            <div className="config-sub-area-control">
                <div className="config-sub-area-control-title">GPU(dml):</div>
                <div className="config-sub-area-control-field">
                    <div className="config-sub-area-buttons left-padding-1">
                        <div onClick={
                            async () => {
                                await serverSetting.updateServerSettings({
                                    ...serverSetting.serverSetting, gpu: serverSetting.serverSetting.gpu == 0 ? -1 : 0
                                })
                            }} className="config-sub-area-button ">{serverSetting.serverSetting.gpu == 0 ? "on" : "off"}</div>
                    </div>
                </div>
            </div >

        ) : (
            <div className="config-sub-area-control">
                <div className="config-sub-area-control-title">GPU:</div>
                <div className="config-sub-area-control-field">
                    <select className="body-select" value={serverSetting.serverSetting.gpu} onChange={(e) => {
                        serverSetting.updateServerSettings({ ...serverSetting.serverSetting, gpu: Number(e.target.value) })
                    }}>
                        {
                            gpusEntry.map(x => {
                                return <option key={x.id} value={x.id}>{x.name}{x.name == "cpu" ? "" : `(${(x.memory / 1024 / 1024 / 1024).toFixed(0)}GB)`} </option>
                            })
                        }
                    </select>
                </div>
            </div>

        )
        return (
            <div className="config-sub-area">
                <div className="config-sub-area-control">
                    <div className="config-sub-area-control-title">CHUNK:</div>
                    <div className="config-sub-area-control-field">
                        <select className="body-select" value={setting.workletNodeSetting.inputChunkNum} onChange={(e) => {
                            setWorkletNodeSetting({ ...setting.workletNodeSetting, inputChunkNum: Number(e.target.value) })
                            trancateBuffer()
                            serverSetting.updateServerSettings({ ...serverSetting.serverSetting, serverReadChunkSize: Number(e.target.value) })
                        }}>
                            {
                                nums.map(x => {
                                    return <option key={x} value={x}>{x} ({(x * 128 * 1000 / 48000).toFixed(1)} ms, {x * 128})</option>
                                })
                            }
                        </select>

                    </div>
                </div>
                <div className="config-sub-area-control">
                    <div className="config-sub-area-control-title">EXTRA:</div>
                    <div className="config-sub-area-control-field">
                        <select className="body-select" value={serverSetting.serverSetting.extraConvertSize} onChange={(e) => {
                            serverSetting.updateServerSettings({ ...serverSetting.serverSetting, extraConvertSize: Number(e.target.value) })
                            trancateBuffer()
                        }}>
                            {
                                [1024 * 4, 1024 * 8, 1024 * 16, 1024 * 32, 1024 * 64, 1024 * 128].map(x => {
                                    return <option key={x} value={x}>{x}</option>
                                })
                            }
                        </select>
                    </div>
                </div>
                {gpuSelect}
            </div>
        )
    }, [serverSetting.serverSetting, setting, serverSetting.updateServerSettings, setWorkletNodeSetting, edition])


    return convertArea
}