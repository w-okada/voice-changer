import React, { useMemo } from "react"
import { useAppRoot } from "../../001_provider/001_AppRootProvider"

export const AnalyzerRow = () => {
    const { appGuiSettingState } = useAppRoot()
    const qualityControlSetting = appGuiSettingState.appGuiSetting.front.qualityControl
    const analyzerRow = useMemo(() => {
        if (!qualityControlSetting.analyzerRow) {
            return <></>
        }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Analyzer(Experimental)</div>
                <div className="body-button-container">
                </div>
            </div>
        )
    }, [])

    return analyzerRow
}