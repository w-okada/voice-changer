export declare const RequestType: {
    readonly voice: "voice";
    readonly config: "config";
    readonly start: "start";
    readonly stop: "stop";
};
export type RequestType = typeof RequestType[keyof typeof RequestType];
export declare const ResponseType: {
    readonly volume: "volume";
    readonly inputData: "inputData";
};
export type ResponseType = typeof ResponseType[keyof typeof ResponseType];
export type VoiceChangerWorkletProcessorRequest = {
    requestType: RequestType;
    voice: ArrayBuffer;
    numTrancateTreshold: number;
    volTrancateThreshold: number;
    volTrancateLength: number;
};
export type VoiceChangerWorkletProcessorResponse = {
    responseType: ResponseType;
    volume?: number;
    recordData?: Float32Array[];
    inputData?: Float32Array;
};
