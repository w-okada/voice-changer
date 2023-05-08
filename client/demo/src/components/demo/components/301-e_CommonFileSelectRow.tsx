import React, { useMemo } from "react"
import { fileSelector } from "@dannadori/voice-changer-client-js"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"

export type CommonFileSelectRowProps = {
    title: string
    acceptExtentions: string[]
    fileKind: Filekinds
}

export const Filekinds = {
    "mmvcv13Config": "mmvcv13Config",
    "mmvcv13Model": "mmvcv13Model",
    "mmvcv15Config": "mmvcv15Config",
    "mmvcv15Model": "mmvcv15Model",
    "ddspSvcModel": "ddspSvcModel",
    "ddspSvcModelConfig": "ddspSvcModelConfig",
    "ddspSvcDiffusion": "ddspSvcDiffusion",
    "ddspSvcDiffusionConfig": "ddspSvcDiffusionConfig",
} as const
export type Filekinds = typeof Filekinds[keyof typeof Filekinds]


export const CommonFileSelectRow = (props: CommonFileSelectRowProps) => {
    const appState = useAppState()
    const guiState = useGuiState()

    const commonFileSelectRow = useMemo(() => {

        const slot = guiState.modelSlotNum

        const getTargetModelData = () => {
            const targetSlot = appState.serverSetting.fileUploadSettings[slot]
            if (!targetSlot) {
                return null
            }
            return targetSlot[props.fileKind]
        }

        const targetModel = getTargetModelData()
        const filenameText = targetModel?.filename || targetModel?.file?.name || ""

        const checkExtention = (filename: string) => {
            const ext = filename.split('.').pop();
            if (!ext) {
                return false
            }
            return props.acceptExtentions.includes(ext)
        }
        const onFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (checkExtention(file.name) == false) {
                alert(`モデルファイルの拡張子は${props.acceptExtentions}である必要があります。`)
                return
            }
            appState.serverSetting.fileUploadSettings[slot][props.fileKind]! = { file: file }
            appState.serverSetting.setFileUploadSetting(slot, {
                ...appState.serverSetting.fileUploadSettings[slot]
            })
        }
        const onFileClearClicked = () => {
            appState.serverSetting.fileUploadSettings[slot][props.fileKind] = null
            appState.serverSetting.setFileUploadSetting(slot, {
                ...appState.serverSetting.fileUploadSettings[slot],
            })
        }


        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-2">{props.title}</div>
                <div className="body-item-text">
                    <div>{filenameText}</div>
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onFileLoadClicked}>select</div>
                    <div className="body-button left-margin-1" onClick={onFileClearClicked}>clear</div>
                </div>
            </div>
        )
    }, [appState.serverSetting.fileUploadSettings, appState.serverSetting.setFileUploadSetting, appState.serverSetting.serverSetting, appState.serverSetting.updateServerSettings, guiState.modelSlotNum])

    return commonFileSelectRow
}