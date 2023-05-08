import React, { useEffect, useMemo, useState } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export type PerformanceRowProps = {
}

export const PerformanceRow = (_props: PerformanceRowProps) => {
    const appState = useAppState()
    const [showPerformanceDetail, setShowPerformanceDetail] = useState<boolean>(false)

    const performanceRow = useMemo(() => {
        if (appState.serverSetting.serverSetting.enableServerAudio) {
            return (
                <div className="body-row split-3-7 left-padding-1 guided">
                    <div className="body-item-title left-padding-1">monitor:</div>
                    <div className="body-item-text">server device mode. refer console.</div>
                </div>
            )
        }
        const performanceDetailLabel = showPerformanceDetail ? "[pre, main, post] <<" : "more >>"
        const performanceData = showPerformanceDetail ? `[${appState.performance.preprocessTime}, ${appState.performance.mainprocessTime},${appState.performance.postprocessTime}]` : ""
        return (
            <>
                <div className="body-row split-3-1-1-1-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-1">monitor:</div>
                    <div className="body-item-text">vol<span className="body-item-text-small">(rms)</span></div>
                    <div className="body-item-text">buf<span className="body-item-text-small">(ms)</span></div>
                    <div className="body-item-text">res<span className="body-item-text-small">(ms)</span></div>
                    <div className="body-item-text">
                        <span onClick={() => { setShowPerformanceDetail(!showPerformanceDetail) }} >{performanceDetailLabel}</span>
                    </div>
                </div>
                <div className="body-row split-3-1-1-1-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-1"></div>
                    <div className="body-item-text">{appState.volume.toFixed(4)}</div>
                    <div className="body-item-text">{appState.bufferingTime}</div>
                    <div className="body-item-text">{appState.performance.responseTime}</div>
                    <div className="body-item-text">{performanceData}</div>
                </div>
            </>
        )
    }, [appState.volume, appState.bufferingTime, appState.performance, showPerformanceDetail, appState.serverSetting.serverSetting.enableServerAudio])

    useEffect(() => {
        if (!appState.updatePerformance) {
            return
        }
        if (appState.serverSetting.serverSetting.enableServerAudio != 1) {
            return
        }
        let execNext = true
        const updatePerformance = async () => {
            await appState.updatePerformance!()
            if (execNext) {
                setTimeout(updatePerformance, 1000 * 2)
            }
        }
        // updatePerformance()
        return () => {
            execNext = false
        }
    }, [appState.updatePerformance, appState.serverSetting.serverSetting.enableServerAudio])

    return (
        <>
            {performanceRow}
        </>
    )
}