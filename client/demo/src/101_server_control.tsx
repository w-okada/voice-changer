import React, { useEffect, useMemo, useState } from "react"
import { useAppState } from "./001_provider/001_AppStateProvider";
import { AnimationTypes, HeaderButton, HeaderButtonProps } from "./components/101_HeaderButton";

export const useServerControl = () => {
    const appState = useAppState()
    const [startWithAudioContextCreate, setStartWithAudioContextCreate] = useState<boolean>(false)

    const accodionButton = useMemo(() => {
        const accodionButtonProps: HeaderButtonProps = {
            stateControlCheckbox: appState.frontendManagerState.stateControls.openServerControlCheckbox,
            tooltip: "Open/Close",
            onIcon: ["fas", "caret-up"],
            offIcon: ["fas", "caret-up"],
            animation: AnimationTypes.spinner,
            tooltipClass: "tooltip-right",
        };
        return <HeaderButton {...accodionButtonProps}></HeaderButton>;
    }, []);

    useEffect(() => {
        if (!startWithAudioContextCreate) {
            return
        }
        appState.frontendManagerState.setIsConverting(true)
        appState.clientSetting.start()
    }, [startWithAudioContextCreate])

    const startButtonRow = useMemo(() => {
        const onStartClicked = async () => {
            if (!appState.audioContext) {
                await new Promise<void>((resolve) => {
                    console.log("wait 2000ms")
                    setTimeout(resolve, 1000 * 2)
                })
                setStartWithAudioContextCreate(true)
            } else {
                appState.frontendManagerState.setIsConverting(true)
                await appState.clientSetting.start()
            }
        }
        const onStopClicked = async () => {
            appState.frontendManagerState.setIsConverting(false)
            await appState.clientSetting.stop()
        }
        const startClassName = appState.frontendManagerState.isConverting ? "body-button-active" : "body-button-stanby"
        const stopClassName = appState.frontendManagerState.isConverting ? "body-button-stanby" : "body-button-active"

        return (
            <div className="body-row split-3-2-2-3 left-padding-1  guided">
                <div className="body-item-title left-padding-1">Start</div>
                <div className="body-button-container">
                    <div onClick={onStartClicked} className={startClassName}>start</div>
                    <div onClick={onStopClicked} className={stopClassName}>stop</div>
                </div>
                <div>
                </div>
                <div className="body-input-container">
                </div>
            </div>
        )
    }, [appState.frontendManagerState.isConverting, appState.clientSetting.start, appState.clientSetting.stop])

    const performanceRow = useMemo(() => {
        return (
            <>
                <div className="body-row split-3-1-1-1-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-1">monitor:</div>
                    <div className="body-item-text">vol<span className="body-item-text-small">(rms)</span></div>
                    <div className="body-item-text">buf<span className="body-item-text-small">(ms)</span></div>
                    <div className="body-item-text">res<span className="body-item-text-small">(ms)</span></div>
                    <div className="body-item-text"></div>
                </div>
                <div className="body-row split-3-1-1-1-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-1"></div>
                    <div className="body-item-text">{appState.volume.toFixed(4)}</div>
                    <div className="body-item-text">{appState.bufferingTime}</div>
                    <div className="body-item-text">{appState.responseTime}</div>
                    <div className="body-item-text"></div>
                </div>
            </>
        )
    }, [appState.volume, appState.bufferingTime, appState.responseTime])



    const infoRow = useMemo(() => {
        const onReloadClicked = async () => {
            const info = await appState.getInfo()
            console.log("info", info)
        }
        return (
            <>
                <div className="body-row split-3-3-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-1">Model Info:</div>
                    <div className="body-item-text">
                        <span className="body-item-text-item">{appState.serverSetting.serverInfo?.configFile || ""}</span>
                        <span className="body-item-text-item">{appState.serverSetting.serverInfo?.pyTorchModelFile || ""}</span>
                        <span className="body-item-text-item">{appState.serverSetting.serverInfo?.onnxModelFile || ""}</span>


                    </div>
                    <div className="body-button-container">
                        <div className="body-button" onClick={onReloadClicked}>reload</div>
                    </div>
                </div>
            </>
        )
    }, [appState.getInfo, appState.serverSetting.serverInfo])

    const serverControl = useMemo(() => {
        return (
            <>
                {appState.frontendManagerState.stateControls.openServerControlCheckbox.trigger}
                <div className="partition">
                    <div className="partition-header">
                        <span className="caret">
                            {accodionButton}
                        </span>
                        <span className="title" onClick={() => { appState.frontendManagerState.stateControls.openServerControlCheckbox.updateState(!appState.frontendManagerState.stateControls.openServerControlCheckbox.checked()) }}>
                            Server Control
                        </span>
                    </div>

                    <div className="partition-content">
                        {startButtonRow}
                        {performanceRow}
                        {infoRow}
                    </div>
                </div>
            </>
        )
    }, [startButtonRow, performanceRow, infoRow])

    return {
        serverControl,
    }
}


