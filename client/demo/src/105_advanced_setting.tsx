import { VoiceChangerMode } from "@dannadori/voice-changer-client-js"
import React, { useMemo, useState } from "react"
import { ClientState } from "./hooks/useClient"


export type UseAdvancedSettingProps = {
    clientState: ClientState
}

export type AdvancedSettingState = {
    advancedSetting: JSX.Element;
}

export const useAdvancedSetting = (props: UseAdvancedSettingProps): AdvancedSettingState => {

    const vfForceDisableRow = useMemo(() => {
        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">VF Disabled</div>
                <div>
                    <input type="checkbox" checked={props.clientState.settingState.vfForceDisabled} onChange={(e) => {
                        props.clientState.setSettingState({
                            ...props.clientState.settingState,
                            vfForceDisabled: e.target.checked
                        })
                    }} />
                </div>
                <div className="body-button-container">
                </div>
            </div>
        )
    }, [props.clientState.settingState])

    const voiceChangeModeRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Voice Change Mode</div>
                <div className="body-select-container">
                    <select className="body-select" value={props.clientState.settingState.voiceChangerMode} onChange={(e) => {
                        props.clientState.setSettingState({
                            ...props.clientState.settingState,
                            voiceChangerMode: e.target.value as VoiceChangerMode
                        })
                    }}>
                        {
                            Object.values(VoiceChangerMode).map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [props.clientState.settingState])

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
    }

}


