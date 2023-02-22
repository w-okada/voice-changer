import React from "react"
import { CorpusSelector } from "./201_CorpusSelector"
import { TextIndexSelector } from "./202_TextIndexSelector"
import { AudioController } from "./203_AudioController"
import { ExportController } from "./204_ExportController"
import { CorpusTextArea } from "./205_CorpusTextArea"
import { WaveSurferView } from "./206_WaveSurferView"

export const Body = () => {

    return (
        <div className="body">
            <div className="body-panel height-10">
                <div className="body-panel-row split-2-2-5-1">
                    <div className="selector-container">
                        <CorpusSelector></CorpusSelector>
                    </div>
                    <div className="pager-container split-3-4-3">
                        <TextIndexSelector></TextIndexSelector>
                    </div>
                    <div className="buttons">
                        <AudioController></AudioController>
                    </div>
                    <div className="buttons">
                        <ExportController></ExportController>
                    </div>
                </div>
            </div>

            <div className="body-panel height-30">
                <div className="body-panel-area">
                    <CorpusTextArea></CorpusTextArea>
                </div>
            </div>
            <div className="body-panel height-60">
                <div className="body-panel-area">
                    <WaveSurferView></WaveSurferView>
                </div>
            </div>
        </div>
    )
}
