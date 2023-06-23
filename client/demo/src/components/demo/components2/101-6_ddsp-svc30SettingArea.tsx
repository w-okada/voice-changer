import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export type DDSPSVC30SettingAreaProps = {
}


export const DDSPSVC30SettingArea = (_props: DDSPSVC30SettingAreaProps) => {
    const { serverSetting } = useAppState()

    const selected = useMemo(() => {
        if (serverSetting.serverSetting.modelSlotIndex == undefined) {
            return
        }
        return serverSetting.serverSetting.modelSlots[serverSetting.serverSetting.modelSlotIndex]
    }, [serverSetting.serverSetting.modelSlotIndex, serverSetting.serverSetting.modelSlots])



    const settingArea = useMemo(() => {
        if (!selected) {
            return <></>
        }

        if (selected.voiceChangerType != "DDSP-SVC") {
            return <></>
        }

        const acc = (
            <div className="character-area-control">
                <div className="character-area-control-title">
                    ACC:
                </div>
                <div className="character-area-control-field">
                    <div className="character-area-slider-control">
                        <span className="character-area-slider-control-kind"></span>
                        <span className="character-area-slider-control-slider">
                            <input type="range" min="1" max="20" step="1" value={serverSetting.serverSetting.diffAcc} onChange={(e) => {
                                serverSetting.updateServerSettings({ ...serverSetting.serverSetting, diffAcc: Number(e.target.value) })
                            }}></input>
                        </span>
                        <span className="character-area-slider-control-val">{serverSetting.serverSetting.diffAcc}</span>
                    </div>

                </div>
            </div>
        )


        const kstep = (
            <div className="character-area-control">
                <div className="character-area-control-title">
                    Kstep:
                </div>
                <div className="character-area-control-field">
                    <div className="character-area-slider-control">
                        <span className="character-area-slider-control-kind"></span>
                        <span className="character-area-slider-control-slider">
                            <input type="range" min="21" max="300" step="1" value={serverSetting.serverSetting.kStep} onChange={(e) => {
                                serverSetting.updateServerSettings({ ...serverSetting.serverSetting, kStep: Number(e.target.value) })
                            }}></input>
                        </span>
                        <span className="character-area-slider-control-val">{serverSetting.serverSetting.kStep}</span>
                    </div>

                </div>
            </div>
        )


        return (
            <>
                {acc}
                {kstep}
            </>
        )
    }, [serverSetting.serverSetting, serverSetting.updateServerSettings, selected])


    return settingArea
}