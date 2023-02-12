import { VoiceChangerWorkletProcessorRequest } from "./@types/voice-changer-worklet-processor";
import { WorkletSetting } from "./const";

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
        const req: VoiceChangerWorkletProcessorRequest = {
            requestType: "voice",
            voice: data,
            numTrancateTreshold: 0,
            volTrancateThreshold: 0,
            volTrancateLength: 0
        }
        this.port.postMessage(req)
    }

    handleMessage(event: any) {
        // console.log(`[Node:handleMessage_] `, event.data.volume);
        if (event.data.responseType === "volume") {
            this.listener.notifyVolume(event.data.volume as number)
        } else if (event.data.responseType === "recordData") {

        } else {
            console.warn(`[worklet_node][voice-changer-worklet-processor] unknown response ${event.data.responseType}`, event.data)
        }
    }

    configure = (setting: WorkletSetting) => {
        const req: VoiceChangerWorkletProcessorRequest = {
            requestType: "config",
            voice: new ArrayBuffer(1),
            numTrancateTreshold: setting.numTrancateTreshold,
            volTrancateThreshold: setting.volTrancateThreshold,
            volTrancateLength: setting.volTrancateLength
        }
        this.port.postMessage(req)
    }

    startOutputRecordingWorklet = () => {
        const req: VoiceChangerWorkletProcessorRequest = {
            requestType: "startRecording",
            voice: new ArrayBuffer(1),
            numTrancateTreshold: 0,
            volTrancateThreshold: 0,
            volTrancateLength: 0
        }
        this.port.postMessage(req)

    }
    stopOutputRecordingWorklet = () => {
        const req: VoiceChangerWorkletProcessorRequest = {
            requestType: "stopRecording",
            voice: new ArrayBuffer(1),
            numTrancateTreshold: 0,
            volTrancateThreshold: 0,
            volTrancateLength: 0
        }
        this.port.postMessage(req)
    }
}