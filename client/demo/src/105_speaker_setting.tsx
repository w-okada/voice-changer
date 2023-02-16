import React, { useEffect, useMemo, useState } from "react"
import { useAppState } from "./001_provider/001_AppStateProvider";
import { AnimationTypes, HeaderButton, HeaderButtonProps } from "./components/101_HeaderButton";

export const useSpeakerSetting = () => {
    const appState = useAppState()
    const accodionButton = useMemo(() => {
        const accodionButtonProps: HeaderButtonProps = {
            stateControlCheckbox: appState.frontendManagerState.stateControls.openSpeakerSettingCheckbox,
            tooltip: "Open/Close",
            onIcon: ["fas", "caret-up"],
            offIcon: ["fas", "caret-up"],
            animation: AnimationTypes.spinner,
            tooltipClass: "tooltip-right",
        };
        return <HeaderButton {...accodionButtonProps}></HeaderButton>;
    }, []);

    const [editSpeakerTargetId, setEditSpeakerTargetId] = useState<number>(0)
    const [editSpeakerTargetName, setEditSpeakerTargetName] = useState<string>("")

    useEffect(() => {
        const src = appState.clientSetting.setting.correspondences?.find(x => {
            return x.sid == appState.serverSetting.setting.srcId
        })
        const dst = appState.clientSetting.setting.correspondences?.find(x => {
            return x.sid == appState.serverSetting.setting.dstId
        })
        const recommendedF0Factor = dst && src ? dst.correspondence / src.correspondence : 0
        appState.serverSetting.setF0Factor(recommendedF0Factor)

    }, [appState.serverSetting.setting.srcId, appState.serverSetting.setting.dstId])

    const srcIdRow = useMemo(() => {
        const selected = appState.clientSetting.setting.correspondences?.find(x => {
            return x.sid == appState.serverSetting.setting.srcId
        })
        return (
            <div className="body-row split-3-2-1-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Source Speaker Id</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.serverSetting.setting.srcId} onChange={(e) => {
                        appState.serverSetting.setSrcId(Number(e.target.value))
                    }}>
                        {
                            // appState.clientSetting.setting.speakers.map(x => {
                            //     return <option key={x.id} value={x.id}>{x.name}({x.id})</option>
                            // })
                            appState.clientSetting.setting.correspondences?.map(x => {
                                return <option key={x.sid} value={x.sid}>{x.dirname}({x.sid})</option>
                            })

                        }
                    </select>
                </div>
                <div className="body-item-text">
                    <div>F0: {selected?.correspondence.toFixed(1) || ""}</div>
                </div>
                <div className="body-item-text"></div>
            </div>
        )
    }, [appState.clientSetting.setting.speakers, appState.serverSetting.setting.srcId, appState.clientSetting.setting.correspondences, appState.serverSetting.setSrcId])

    const dstIdRow = useMemo(() => {
        const selected = appState.clientSetting.setting.correspondences?.find(x => {
            return x.sid == appState.serverSetting.setting.dstId
        })
        return (
            <div className="body-row split-3-2-1-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Destination Speaker Id</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.serverSetting.setting.dstId} onChange={(e) => {
                        appState.serverSetting.setDstId(Number(e.target.value))
                    }}>
                        {
                            // appState.clientSetting.setting.speakers.map(x => {
                            //     return <option key={x.id} value={x.id}>{x.name}({x.id})</option>
                            // })
                            appState.clientSetting.setting.correspondences?.map(x => {
                                return <option key={x.sid} value={x.sid}>{x.dirname}({x.sid})</option>
                            })
                        }
                    </select>
                </div>
                <div className="body-item-text">
                    <div>F0: {selected?.correspondence.toFixed(1) || ""}</div>
                </div>
                <div className="body-item-text"></div>
            </div>
        )
    }, [appState.clientSetting.setting.speakers, appState.serverSetting.setting.dstId, appState.clientSetting.setting.correspondences, appState.serverSetting.setDstId])

    const editSpeakerIdMappingRow = useMemo(() => {
        const onSetSpeakerMappingClicked = async () => {
            const targetId = editSpeakerTargetId
            const targetName = editSpeakerTargetName
            const targetSpeaker = appState.clientSetting.setting.speakers.find(x => { return x.id == targetId })
            if (targetSpeaker) {
                if (targetName.length == 0) { // Delete
                    const newSpeakers = appState.clientSetting.setting.speakers.filter(x => { return x.id != targetId })
                    appState.clientSetting.setSpeakers(newSpeakers)
                } else { // Update
                    targetSpeaker.name = targetName
                    appState.clientSetting.setSpeakers([...appState.clientSetting.setting.speakers])
                }
            } else {
                if (targetName.length == 0) { // Noop
                } else {// add
                    appState.clientSetting.setting.speakers.push({
                        id: targetId,
                        name: targetName
                    })
                    appState.clientSetting.setSpeakers([...appState.clientSetting.setting.speakers])
                }
            }
        }
        return (
            <div className="body-row split-3-1-2-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Edit Speaker Mapping</div>
                <div className="body-input-container">
                    <input type="number" min={1} max={256} step={1} value={editSpeakerTargetId} onChange={(e) => {
                        const id = Number(e.target.value)
                        setEditSpeakerTargetId(id)
                        setEditSpeakerTargetName(appState.clientSetting.setting.speakers.find(x => { return x.id == id })?.name || "")
                    }} />
                </div>
                <div className="body-input-container">
                    <input type="text" value={editSpeakerTargetName} onChange={(e) => {
                        setEditSpeakerTargetName(e.target.value)
                    }} />
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onSetSpeakerMappingClicked}>set</div>
                </div>
            </div>
        )
    }, [appState.clientSetting.setting.speakers, editSpeakerTargetId, editSpeakerTargetName])


    const f0FactorRow = useMemo(() => {
        const src = appState.clientSetting.setting.correspondences?.find(x => {
            return x.sid == appState.serverSetting.setting.srcId
        })
        const dst = appState.clientSetting.setting.correspondences?.find(x => {
            return x.sid == appState.serverSetting.setting.dstId
        })

        const recommendedF0Factor = dst && src ? dst.correspondence / src.correspondence : 0

        return (
            <div className="body-row split-3-2-1-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">F0 Factor</div>
                <div className="body-input-container">
                    <input type="range" className="body-item-input-slider" min="0.1" max="5.0" step="0.1" value={appState.serverSetting.setting.f0Factor} onChange={(e) => {
                        appState.serverSetting.setF0Factor(Number(e.target.value))
                    }}></input>
                    <span className="body-item-input-slider-val">{appState.serverSetting.setting.f0Factor.toFixed(1)}</span>
                </div>
                <div className="body-item-text"></div>
                <div className="body-item-text">recommend: {recommendedF0Factor.toFixed(1)}</div>
            </div>
        )
    }, [appState.serverSetting.setting.f0Factor, appState.serverSetting.setting.srcId, appState.serverSetting.setting.dstId, appState.clientSetting.setting.correspondences, appState.serverSetting.setF0Factor])

    const speakerSetting = useMemo(() => {
        return (
            <>
                {appState.frontendManagerState.stateControls.openSpeakerSettingCheckbox.trigger}
                <div className="partition">
                    <div className="partition-header">
                        <span className="caret">
                            {accodionButton}
                        </span>
                        <span className="title" onClick={() => { appState.frontendManagerState.stateControls.openSpeakerSettingCheckbox.updateState(!appState.frontendManagerState.stateControls.openSpeakerSettingCheckbox.checked()) }}>
                            Speaker Setting
                        </span>
                    </div>

                    <div className="partition-content">
                        {srcIdRow}
                        {dstIdRow}
                        {f0FactorRow}
                    </div>
                </div>
            </>
        )
    }, [srcIdRow, dstIdRow, editSpeakerIdMappingRow, f0FactorRow])

    return {
        speakerSetting,
    }

}

