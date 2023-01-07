import { VoiceChangerMode } from "@dannadori/voice-changer-client-js"
import React, { useMemo, useState } from "react"

export type AdvancedSettingState = {
    advancedSetting: JSX.Element;
    vfForceDisabled: boolean;
    voiceChangeMode: VoiceChangerMode;
}


export const useAdvancedSetting = (): AdvancedSettingState => {

    const [vfForceDisabled, setVfForceDisabled] = useState<boolean>(false)
    const [voiceChangeMode, setVoiceChangeMode] = useState<VoiceChangerMode>("realtime")

    const vfForceDisableRow = useMemo(() => {
        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">VF Disabled</div>
                <div>
                    <input type="checkbox" checked={vfForceDisabled} onChange={(e) => setVfForceDisabled(e.target.checked)} />
                </div>
                <div className="body-button-container">
                </div>
            </div>
        )
    }, [vfForceDisabled])

    const voiceChangeModeRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Voice Change Mode</div>
                <div className="body-select-container">
                    <select className="body-select" value={voiceChangeMode} onChange={(e) => { setVoiceChangeMode(e.target.value as VoiceChangerMode) }}>
                        {
                            Object.values(VoiceChangerMode).map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [])

    const advancedSetting = useMemo(() => {
        return (
            <>
                <div className="body-row split-3-7 left-padding-1">
                    <div className="body-sub-section-title">Advanced Setting</div>
                    <div className="body-select-container">
                    </div>
                </div>
                {vfForceDisableRow}
                {voiceChangeModeRow}
            </>
        )
    }, [vfForceDisableRow, voiceChangeModeRow])

    return {
        advancedSetting,
        vfForceDisabled,
        voiceChangeMode,
    }

}


