export type VolumeListener = {
    notifyVolume: (vol: number) => void
}

export class VoiceChangerWorkletNode extends AudioWorkletNode {
    private listener: VolumeListener
    constructor(context: AudioContext, listener: VolumeListener) {
        super(context, "voice-changer-worklet-processor");
        this.port.onmessage = this.handleMessage.bind(this);
        this.listener = listener
        console.log(`[worklet_node][voice-changer-worklet-processor] created.`);
    }

    postReceivedVoice = (data: ArrayBuffer) => {
        this.port.postMessage({
            data: data,
        }, [data]);
    }

    handleMessage(event: any) {
        // console.log(`[Node:handleMessage_] `, event.data.volume);
        this.listener.notifyVolume(event.data.volume as number)
    }
}