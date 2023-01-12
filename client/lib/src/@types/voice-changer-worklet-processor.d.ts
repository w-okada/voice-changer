export declare const RequestType: {
    readonly voice: "voice";
    readonly config: "config";
};
export type RequestType = typeof RequestType[keyof typeof RequestType];
export type VoiceChangerWorkletProcessorRequest = {
    requestType: RequestType;
    voice: ArrayBuffer;
    numTrancateTreshold: number;
    volTrancateThreshold: number;
    volTrancateLength: number;
};
