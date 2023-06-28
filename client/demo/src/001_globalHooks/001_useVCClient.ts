import { ClientState, useClient } from "@dannadori/voice-changer-client-js"

export type UseVCClientProps = {
    audioContext: AudioContext | null
}

export type VCClientState = {
    clientState: ClientState
}

export const useVCClient = (props: UseVCClientProps): VCClientState => {
    const clientState = useClient({
        audioContext: props.audioContext
    })
    console.log("useVCClient", props.audioContext)

    const ret: VCClientState = {
        clientState
    }


    return ret

}