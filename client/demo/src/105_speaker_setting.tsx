import React, { useEffect, useMemo } from "react"
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


    const calcDefaultF0Factor = (srcId: number, dstId: number) => {
        const src = appState.clientSetting.clientSetting.correspondences?.find(x => {
            return x.sid == srcId
        })
        const dst = appState.clientSetting.clientSetting.correspondences?.find(x => {
            return x.sid == dstId
        })
        console.log("calcDefaultF0Factor", srcId, dstId, src, dst)
        const recommendedF0Factor = dst && src ? dst.correspondence / src.correspondence : 0
        return recommendedF0Factor
    }
    useEffect(() => {
        const recF0 = calcDefaultF0Factor(appState.serverSetting.serverSetting.srcId, appState.serverSetting.serverSetting.dstId)
        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, f0Factor: recF0 })
    }, [appState.clientSetting.clientSetting.correspondences])


    const srcIdRow = useMemo(() => {
        const selected = appState.clientSetting.clientSetting.correspondences?.find(x => {
            return x.sid == appState.serverSetting.serverSetting.srcId
        })
        return (
            <div className="body-row split-3-2-1-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Source Speaker Id</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.serverSetting.serverSetting.srcId} onChange={(e) => {
                        const recF0 = calcDefaultF0Factor(Number(e.target.value), appState.serverSetting.serverSetting.dstId)
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, srcId: Number(e.target.value), f0Factor: recF0 })
                    }}>
                        {
                            appState.clientSetting.clientSetting.correspondences?.map(x => {
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
    }, [appState.serverSetting.serverSetting.srcId, appState.serverSetting.serverSetting.dstId, appState.clientSetting.clientSetting.correspondences, appState.serverSetting.updateServerSettings])

    const dstIdRow = useMemo(() => {
        const selected = appState.clientSetting.clientSetting.correspondences?.find(x => {
            return x.sid == appState.serverSetting.serverSetting.dstId
        })
        return (
            <div className="body-row split-3-2-1-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Destination Speaker Id</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.serverSetting.serverSetting.dstId} onChange={(e) => {
                        const recF0 = calcDefaultF0Factor(appState.serverSetting.serverSetting.srcId, Number(e.target.value))
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, dstId: Number(e.target.value), f0Factor: recF0 })

                    }}>
                        {
                            // appState.clientSetting.setting.speakers.map(x => {
                            //     return <option key={x.id} value={x.id}>{x.name}({x.id})</option>
                            // })
                            appState.clientSetting.clientSetting.correspondences?.map(x => {
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
    }, [appState.serverSetting.serverSetting.srcId, appState.serverSetting.serverSetting.dstId, appState.clientSetting.clientSetting.correspondences, appState.serverSetting.updateServerSettings])

    // const editSpeakerIdMappingRow = useMemo(() => {
    //     const onSetSpeakerMappingClicked = async () => {
    //         const targetId = editSpeakerTargetId
    //         const targetName = editSpeakerTargetName
    //         const targetSpeaker = appState.clientSetting.setting.speakers.find(x => { return x.id == targetId })
    //         if (targetSpeaker) {
    //             if (targetName.length == 0) { // Delete
    //                 const newSpeakers = appState.clientSetting.setting.speakers.filter(x => { return x.id != targetId })
    //                 appState.clientSetting.setSpeakers(newSpeakers)
    //             } else { // Update
    //                 targetSpeaker.name = targetName
    //                 appState.clientSetting.setSpeakers([...appState.clientSetting.setting.speakers])
    //             }
    //         } else {
    //             if (targetName.length == 0) { // Noop
    //             } else {// add
    //                 appState.clientSetting.setting.speakers.push({
    //                     id: targetId,
    //                     name: targetName
    //                 })
    //                 appState.clientSetting.setSpeakers([...appState.clientSetting.setting.speakers])
    //             }
    //         }
    //     }
    //     return (
    //         <div className="body-row split-3-1-2-4 left-padding-1 guided">
    //             <div className="body-item-title left-padding-1">Edit Speaker Mapping</div>
    //             <div className="body-input-container">
    //                 <input type="number" min={1} max={256} step={1} value={editSpeakerTargetId} onChange={(e) => {
    //                     const id = Number(e.target.value)
    //                     setEditSpeakerTargetId(id)
    //                     setEditSpeakerTargetName(appState.clientSetting.setting.speakers.find(x => { return x.id == id })?.name || "")
    //                 }} />
    //             </div>
    //             <div className="body-input-container">
    //                 <input type="text" value={editSpeakerTargetName} onChange={(e) => {
    //                     setEditSpeakerTargetName(e.target.value)
    //                 }} />
    //             </div>
    //             <div className="body-button-container">
    //                 <div className="body-button" onClick={onSetSpeakerMappingClicked}>set</div>
    //             </div>
    //         </div>
    //     )
    // }, [appState.clientSetting.setting.speakers, editSpeakerTargetId, editSpeakerTargetName])


    const f0FactorRow = useMemo(() => {
        const src = appState.clientSetting.clientSetting.correspondences?.find(x => {
            return x.sid == appState.serverSetting.serverSetting.srcId
        })
        const dst = appState.clientSetting.clientSetting.correspondences?.find(x => {
            return x.sid == appState.serverSetting.serverSetting.dstId
        })

        const recommendedF0Factor = dst && src ? dst.correspondence / src.correspondence : 0

        return (
            <div className="body-row split-3-2-1-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">F0 Factor</div>
                <div className="body-input-container">
                    <input type="range" className="body-item-input-slider" min="0.1" max="5.0" step="0.1" value={appState.serverSetting.serverSetting.f0Factor} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, f0Factor: Number(e.target.value) })
                    }}></input>
                    <span className="body-item-input-slider-val">{appState.serverSetting.serverSetting.f0Factor.toFixed(1)}</span>
                </div>
                <div className="body-item-text"></div>
                <div className="body-item-text">recommend: {recommendedF0Factor.toFixed(1)}</div>
            </div>
        )
    }, [appState.serverSetting.serverSetting.f0Factor, appState.serverSetting.serverSetting.srcId, appState.serverSetting.serverSetting.dstId, appState.clientSetting.clientSetting.correspondences, appState.serverSetting.updateServerSettings])

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
    }, [srcIdRow, dstIdRow, f0FactorRow])

    return {
        speakerSetting,
    }

}


