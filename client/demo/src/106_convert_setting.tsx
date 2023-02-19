import React, { useMemo } from "react"
import { useAppState } from "./001_provider/001_AppStateProvider";
import { AnimationTypes, HeaderButton, HeaderButtonProps } from "./components/101_HeaderButton";


export type ConvertSettingState = {
    convertSetting: JSX.Element;
}

export const useConvertSetting = (): ConvertSettingState => {
    const appState = useAppState()
    const accodionButton = useMemo(() => {
        const accodionButtonProps: HeaderButtonProps = {
            stateControlCheckbox: appState.frontendManagerState.stateControls.openConverterSettingCheckbox,
            tooltip: "Open/Close",
            onIcon: ["fas", "caret-up"],
            offIcon: ["fas", "caret-up"],
            animation: AnimationTypes.spinner,
            tooltipClass: "tooltip-right",
        };
        return <HeaderButton {...accodionButtonProps}></HeaderButton>;
    }, []);
    const inputChunkNumRow = useMemo(() => {
        return (
            <div className="body-row split-3-2-1-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Input Chunk Num(128sample/chunk)</div>
                <div className="body-input-container">
                    <input type="number" min={1} max={256} step={1} value={appState.workletNodeSetting.workletNodeSetting.inputChunkNum} onChange={(e) => {
                        appState.workletNodeSetting.updateWorkletNodeSetting({ ...appState.workletNodeSetting.workletNodeSetting, inputChunkNum: Number(e.target.value) })
                    }} />
                </div>
                <div className="body-item-text">
                    <div>buff: {(appState.workletNodeSetting.workletNodeSetting.inputChunkNum * 128 * 1000 / 48000).toFixed(1)}ms</div>
                </div>
                <div className="body-item-text"></div>

            </div>
        )
    }, [appState.workletNodeSetting.workletNodeSetting.inputChunkNum, appState.workletNodeSetting.updateWorkletNodeSetting])

    const gpuRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title  left-padding-1">GPU</div>
                <div className="body-input-container">
                    <input type="number" min={-2} max={5} step={1} value={appState.serverSetting.serverSetting.gpu} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, gpu: Number(e.target.value) })
                    }} />
                </div>
            </div>
        )
    }, [appState.serverSetting.serverSetting.gpu, appState.serverSetting.updateServerSettings])


    const convertSetting = useMemo(() => {
        return (
            <>
                {appState.frontendManagerState.stateControls.openConverterSettingCheckbox.trigger}
                <div className="partition">
                    <div className="partition-header">
                        <span className="caret">
                            {accodionButton}
                        </span>
                        <span className="title" onClick={() => { appState.frontendManagerState.stateControls.openConverterSettingCheckbox.updateState(!appState.frontendManagerState.stateControls.openConverterSettingCheckbox.checked()) }}>
                            Converter Setting
                        </span>
                    </div>

                    <div className="partition-content">
                        {inputChunkNumRow}
                        {gpuRow}
                    </div>
                </div>
            </>
        )
    }, [inputChunkNumRow, gpuRow])

    return {
        convertSetting,
    }

}


