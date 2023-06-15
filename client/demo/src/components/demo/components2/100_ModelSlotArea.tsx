import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"
import { useMessageBuilder } from "../../../hooks/useMessageBuilder"

export type ModelSlotAreaProps = {
}


export const ModelSlotArea = (_props: ModelSlotAreaProps) => {
    const { serverSetting, getInfo } = useAppState()
    const guiState = useGuiState()
    const messageBuilderState = useMessageBuilder()

    useMemo(() => {
        messageBuilderState.setMessage(__filename, "edit", { "ja": "編集", "en": "edit" })
    }, [])


    const modelTiles = useMemo(() => {
        if (!serverSetting.serverSetting.slotInfos) {
            return []
        }
        return serverSetting.serverSetting.slotInfos.map((x, index) => {
            if (x.voiceChangerType == null) {
                return null
            }
            if (x.modelFile.length == 0) {
                return null
            }
            const tileContainerClass = index == serverSetting.serverSetting.slotIndex ? "model-slot-tile-container-selected" : "model-slot-tile-container"
            const name = x.name.length > 8 ? x.name.substring(0, 7) + "..." : x.name
            const iconElem = x.iconFile.length > 0 ?
                <img className="model-slot-tile-icon" src={x.iconFile} alt={x.name} /> :
                <div className="model-slot-tile-icon-no-entry">no image</div>

            const clickAction = async () => {
                const dummyModelSlotIndex = (Math.floor(Date.now() / 1000)) * 1000 + index
                await serverSetting.updateServerSettings({ ...serverSetting.serverSetting, slotIndex: dummyModelSlotIndex })
                setTimeout(() => { // quick hack
                    getInfo()
                }, 1000 * 2)
            }

            return (
                <div key={index} className={tileContainerClass} onClick={clickAction}>
                    <div className="model-slot-tile-icon-div">
                        {iconElem}
                    </div>
                    <div className="model-slot-tile-dscription">
                        {name}
                    </div>
                </div >
            )
        }).filter(x => x != null)
    }, [serverSetting.serverSetting.slotInfos, serverSetting.serverSetting.slotIndex])


    const modelSlotArea = useMemo(() => {
        const onModelSlotEditClicked = () => {
            guiState.stateControls.showModelSlotManagerCheckbox.updateState(true)
        }
        return (
            <div className="model-slot-area">
                <div className="model-slot-panel">
                    <div className="model-slot-tiles-container">{modelTiles}</div>
                    <div className="model-slot-buttons">
                        <div className="model-slot-button" onClick={onModelSlotEditClicked}>
                            {messageBuilderState.getMessage(__filename, "edit")}
                        </div>
                    </div>

                </div>
            </div>
        )
    }, [modelTiles])

    return modelSlotArea
}