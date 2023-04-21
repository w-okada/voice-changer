import { ClientState, useClient, ClientType } from "@dannadori/voice-changer-client-js"

export type UseVCClientProps = {
    audioContext: AudioContext | null
    clientType: ClientType | null
}

export type VCClientState = {
    clientState: ClientState
}

export const useVCClient = (props: UseVCClientProps): VCClientState => {
    const clientState = useClient({
        audioContext: props.audioContext,
        clientType: props.clientType,

    })


    // const setClientType = (clientType: ClientType) => {
    //     console.log("SET CLIENT TYPE", clientType)
    //     clientState.setClientType(clientType)
    // }

    const ret: VCClientState = {
        clientState
    }


    return ret

}