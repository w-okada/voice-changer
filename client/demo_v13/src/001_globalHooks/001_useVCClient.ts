import { ClientState, useClient, ClientType } from "@dannadori/voice-changer-client-js"
import { AUDIO_ELEMENT_FOR_PLAY_RESULT } from "../const"

export type UseVCClientProps = {
    audioContext: AudioContext | null
    clientType: ClientType
}

export type VCClientState = {
    clientState: ClientState
}

export const useVCClient = (props: UseVCClientProps) => {
    const clientState = useClient({
        clientType: props.clientType,
        audioContext: props.audioContext,
        audioOutputElementId: AUDIO_ELEMENT_FOR_PLAY_RESULT
    })

    const ret: VCClientState = {
        clientState
    }

    return ret

}