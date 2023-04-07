import { ClientState, useClient, ClientType } from "@dannadori/voice-changer-client-js"

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
    })

    const ret: VCClientState = {
        clientState
    }

    return ret

}