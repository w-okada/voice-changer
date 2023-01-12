import { BufferSize } from "@dannadori/voice-changer-client-js"
import React, { useMemo, useState } from "react"
import { ClientState } from "./hooks/useClient"

export type UseConvertSettingProps = {
    clientState: ClientState
}

export type ConvertSettingState = {
    convertSetting: JSX.Element;
}

export const useConvertSetting = (props: UseConvertSettingProps): ConvertSettingState => {

    const inputChunkNumRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Input Chunk Num(128sample/chunk)</div>
                <div className="body-input-container">
                    <input type="number" min={1} max={256} step={1} value={props.clientState.clientSetting.setting.inputChunkNum} onChange={(e) => {
                        props.clientState.clientSetting.setInputChunkNum(Number(e.target.value))
                    }} />
                </div>
            </div>
        )
    }, [props.clientState.clientSetting.setting.inputChunkNum, props.clientState.clientSetting.setInputChunkNum])

    const gpuRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title  left-padding-1">GPU</div>
                <div className="body-input-container">
                    <input type="number" min={-2} max={5} step={1} value={props.clientState.serverSetting.setting.gpu} onChange={(e) => {
                        props.clientState.serverSetting.setGpu(Number(e.target.value))
                    }} />
                </div>
            </div>
        )
    }, [props.clientState.serverSetting.setting.gpu, props.clientState.serverSetting.setGpu])


    const convertSetting = useMemo(() => {
        return (
            <>
                <div className="body-row split-3-7 left-padding-1">
                    <div className="body-sub-section-title">Converter Setting</div>
                    <div className="body-select-container">
                    </div>
                </div>
                {inputChunkNumRow}
                {gpuRow}

            </>
        )
    }, [inputChunkNumRow, gpuRow])

    return {
        convertSetting,
    }

}


