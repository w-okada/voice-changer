import { ClientState, useClient } from "@dannadori/voice-changer-client-js"
import { useEffect, useState } from "react"
import { AUDIO_ELEMENT_FOR_PLAY_RESULT } from "../const"

export type UseVCClientProps = {
    audioContext: AudioContext
}

export type VCClientState = {
    audioContext: AudioContext
    clientState: ClientState
}

export const useVCClient = (props: UseVCClientProps) => {

    const clientState = useClient({
        audioContext: props.audioContext,
        audioOutputElementId: AUDIO_ELEMENT_FOR_PLAY_RESULT
    })

    const ret: VCClientState = {
        audioContext: props.audioContext,
        clientState
    }

    return ret

}