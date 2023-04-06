import React, { useMemo, useState } from "react"
import { useAppState } from "./001_provider/001_AppStateProvider";
import { AnimationTypes, HeaderButton, HeaderButtonProps } from "./components/101_HeaderButton";

export const useSpeakerSetting = () => {
    const appState = useAppState()
    const [editSpeakerTargetId, setEditSpeakerTargetId] = useState<number>(0)
    const [editSpeakerTargetName, setEditSpeakerTargetName] = useState<string>("")


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


    // const calcDefaultF0Factor = (srcId: number, dstId: number) => {
    //     const src = appState.clientSetting.clientSetting.correspondences?.find(x => {
    //         return x.sid == srcId
    //     })
    //     const dst = appState.clientSetting.clientSetting.correspondences?.find(x => {
    //         return x.sid == dstId
    //     })
    //     const recommendedF0Factor = dst && src ? dst.correspondence / src.correspondence : 0
    //     return recommendedF0Factor
    // }
    // useEffect(() => {
    //     const recF0 = calcDefaultF0Factor(appState.serverSetting.serverSetting.srcId, appState.serverSetting.serverSetting.dstId)
    //     appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, f0Factor: recF0 })
    // }, [appState.clientSetting.clientSetting.correspondences])


    const srcIdRow = useMemo(() => {
        return (
            <div className="body-row split-3-2-1-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Source Speaker Id</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.serverSetting.serverSetting.srcId} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, srcId: Number(e.target.value) })
                    }}>
                        {
                            appState.clientSetting.clientSetting.speakers.map(x => {
                                return <option key={x.id} value={x.id}>{x.name}({x.id})</option>
                            })

                        }
                    </select>
                </div>
                <div className="body-item-text">
                </div>
                <div className="body-item-text"></div>
            </div>
        )
    }, [appState.serverSetting.serverSetting.srcId, appState.serverSetting.serverSetting.dstId, appState.clientSetting.clientSetting.speakers, appState.serverSetting.updateServerSettings])

    const dstIdRow = useMemo(() => {
        return (
            <div className="body-row split-3-2-1-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Destination Speaker Id</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.serverSetting.serverSetting.dstId} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, dstId: Number(e.target.value) })

                    }}>
                        {
                            appState.clientSetting.clientSetting.speakers?.map(x => {
                                return <option key={x.id} value={x.id}>{x.name}({x.id})</option>
                            })
                        }
                    </select>
                </div>
                <div className="body-item-text">
                </div>
                <div className="body-item-text"></div>
            </div>
        )
    }, [appState.serverSetting.serverSetting.srcId, appState.serverSetting.serverSetting.dstId, appState.clientSetting.clientSetting.speakers, appState.serverSetting.updateServerSettings])

    const editSpeakerIdMappingRow = useMemo(() => {
        const onSetSpeakerMappingClicked = async () => {
            const targetId = editSpeakerTargetId
            const targetName = editSpeakerTargetName
            const targetSpeaker = appState.clientSetting.clientSetting.speakers.find(x => { return x.id == targetId })
            if (targetSpeaker) {
                if (targetName.length == 0) { // Delete
                    const newSpeakers = appState.clientSetting.clientSetting.speakers.filter(x => { return x.id != targetId })
                    appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, speakers: newSpeakers })
                } else { // Update
                    targetSpeaker.name = targetName
                    appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, speakers: [...appState.clientSetting.clientSetting.speakers] })
                }
            } else {
                if (targetName.length == 0) { // Noop
                } else {// add
                    appState.clientSetting.clientSetting.speakers.push({
                        id: targetId,
                        name: targetName
                    })
                    appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, speakers: [...appState.clientSetting.clientSetting.speakers] })

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
                        setEditSpeakerTargetName(appState.clientSetting.clientSetting.speakers.find(x => { return x.id == id })?.name || "")
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
    }, [appState.clientSetting.clientSetting.speakers, editSpeakerTargetId, editSpeakerTargetName])



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
                        {editSpeakerIdMappingRow}
                    </div>
                </div>
            </>
        )
    }, [srcIdRow, dstIdRow, editSpeakerIdMappingRow])

    return {
        speakerSetting,
    }

}


