export class VoiceChangerWorkletNode extends AudioWorkletNode {
    constructor(context: AudioContext) {
        super(context, "voice-changer-worklet-processor");
        this.port.onmessage = this.handleMessage.bind(this);
        console.log(`[worklet_node][voice-changer-worklet-processor] created.`);
    }

    postReceivedVoice = (data: ArrayBuffer) => {
        this.port.postMessage({
            data: data,
        }, [data]);
    }

    handleMessage(event: any) {
        console.log(`[Node:handleMessage_] `, event.data.volume);
    }
}