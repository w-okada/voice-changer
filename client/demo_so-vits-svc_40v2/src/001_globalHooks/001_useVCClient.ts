import { ClientState, useClient } from "@dannadori/voice-changer-client-js"
import { AUDIO_ELEMENT_FOR_PLAY_RESULT, CLIENT_TYPE } from "../const"

export type UseVCClientProps = {
    audioContext: AudioContext | null
}

export type VCClientState = {
    clientState: ClientState
}

export const useVCClient = (props: UseVCClientProps) => {
    const clientState = useClient({
        clientType: CLIENT_TYPE,
        audioContext: props.audioContext,
        audioOutputElementId: AUDIO_ELEMENT_FOR_PLAY_RESULT
    })

    const ret: VCClientState = {
        clientState
    }

    return ret

}