import React, { useMemo, useEffect, useRef } from "react"
import { fileSelectorAsDataURL } from "@dannadori/voice-changer-client-js"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"
import { AUDIO_ELEMENT_FOR_TEST_CONVERTED, AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK, AUDIO_ELEMENT_FOR_TEST_ORIGINAL } from "../../../const"

export const AudioInputMediaRow = () => {
    const appState = useAppState()
    const guiState = useGuiState()
    const audioSrcNode = useRef<MediaElementAudioSourceNode>()

    useEffect(() => {
        [AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK].forEach(x => {
            const audio = document.getElementById(x) as HTMLAudioElement
            if (audio) {
                audio.volume = guiState.fileInputEchoback ? 1 : 0
            }
        })
    }, [guiState.fileInputEchoback])

    const audioInputMediaRow = useMemo(() => {
        if (guiState.audioInputForGUI != "file") {
            return <></>
        }

        const onFileLoadClicked = async () => {
            const url = await fileSelectorAsDataURL("")

            // input stream for client.
            const audio = document.getElementById(AUDIO_ELEMENT_FOR_TEST_CONVERTED) as HTMLAudioElement
            audio.pause()
            audio.srcObject = null
            audio.src = url
            await audio.play()
            if (!audioSrcNode.current) {
                audioSrcNode.current = appState.audioContext!.createMediaElementSource(audio);
            }
            if (audioSrcNode.current.mediaElement != audio) {
                audioSrcNode.current = appState.audioContext!.createMediaElementSource(audio);
            }

            const dst = appState.audioContext.createMediaStreamDestination()
            audioSrcNode.current.connect(dst)
            try {
                appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, audioInput: dst.stream })
            } catch (e) {
                console.error(e)
            }

            const audio_echo = document.getElementById(AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK) as HTMLAudioElement
            audio_echo.srcObject = dst.stream
            audio_echo.play()
            audio_echo.volume = 0
            guiState.setFileInputEchoback(false)

            // original stream to play.
            const audio_org = document.getElementById(AUDIO_ELEMENT_FOR_TEST_ORIGINAL) as HTMLAudioElement
            audio_org.src = url
            audio_org.pause()

            // audio_org.onplay = () => {
            //     console.log(audioOutputRef.current)
            //     // @ts-ignore
            //     audio_org.setSinkId(audioOutputRef.current)
            // }
        }

        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title"></div>
                <div className="body-item-text">
                    <div style={{ display: "none" }}>
                        org:<audio id={AUDIO_ELEMENT_FOR_TEST_ORIGINAL} controls></audio>
                    </div>
                    <div>
                        <audio id={AUDIO_ELEMENT_FOR_TEST_CONVERTED} controls></audio>
                        <audio id={AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK} controls hidden></audio>
                    </div>
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onFileLoadClicked}>load</div>
                    <input type="checkbox" checked={guiState.fileInputEchoback} onChange={(e) => { guiState.setFileInputEchoback(e.target.checked) }} /> echoback
                </div>
            </div>
        )
    }, [guiState.audioInputForGUI, guiState.fileInputEchoback])

    return audioInputMediaRow
}