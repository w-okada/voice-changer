import { DefaultVoiceChangerRequestParamas, DefaultVoiceChangerOptions, Speaker } from "@dannadori/voice-changer-client-js"
import React, { useMemo, useState } from "react"


export const useSpeakerSetting = () => {
    const [speakers, setSpeakers] = useState<Speaker[]>(DefaultVoiceChangerOptions.speakers)
    const [editSpeakerTargetId, setEditSpeakerTargetId] = useState<number>(0)
    const [editSpeakerTargetName, setEditSpeakerTargetName] = useState<string>("")

    const [srcId, setSrcId] = useState<number>(DefaultVoiceChangerRequestParamas.srcId)
    const [dstId, setDstId] = useState<number>(DefaultVoiceChangerRequestParamas.dstId)


    const srcIdRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Source Speaker Id</div>
                <div className="body-select-container">
                    <select className="body-select" value={srcId} onChange={(e) => { setSrcId(Number(e.target.value)) }}>
                        {
                            speakers.map(x => {
                                return <option key={x.id} value={x.id}>{x.name}({x.id})</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [srcId, speakers])

    const dstIdRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Destination Speaker Id</div>
                <div className="body-select-container">
                    <select className="body-select" value={dstId} onChange={(e) => { setDstId(Number(e.target.value)) }}>
                        {
                            speakers.map(x => {
                                return <option key={x.id} value={x.id}>{x.name}({x.id})</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [dstId, speakers])

    const editSpeakerIdMappingRow = useMemo(() => {
        const onSetSpeakerMappingClicked = async () => {
            const targetId = editSpeakerTargetId
            const targetName = editSpeakerTargetName
            const targetSpeaker = speakers.find(x => { return x.id == targetId })
            if (targetSpeaker) {
                if (targetName.length == 0) { // Delete
                    const newSpeakers = speakers.filter(x => { return x.id != targetId })
                    setSpeakers(newSpeakers)
                } else { // Update
                    targetSpeaker.name = targetName
                    setSpeakers([...speakers])
                }
            } else {
                if (targetName.length == 0) { // Noop
                } else {// add
                    speakers.push({
                        id: targetId,
                        name: targetName
                    })
                    setSpeakers([...speakers])
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
                        setEditSpeakerTargetName(speakers.find(x => { return x.id == id })?.name || "")
                    }} />
                </div>
                <div className="body-input-container">
                    <input type="text" value={editSpeakerTargetName} onChange={(e) => { setEditSpeakerTargetName(e.target.value) }} />
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onSetSpeakerMappingClicked}>set</div>
                </div>
            </div>

        )
    }, [speakers])


    const speakerSetting = useMemo(() => {
        return (
            <>
                <div className="body-row split-3-7 left-padding-1">
                    <div className="body-sub-section-title">Speaker Setting</div>
                    <div className="body-select-container">
                    </div>
                </div>
                {srcIdRow}
                {dstIdRow}
                {editSpeakerIdMappingRow}
            </>
        )
    }, [srcIdRow, dstIdRow, editSpeakerIdMappingRow])

    return {
        speakerSetting,
        srcId,
        dstId,
    }

}


