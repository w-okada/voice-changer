import { MergeModelRequest, ServerSettingKey } from "../const";
import { ServerRestClient } from "./ServerRestClient";

export class ServerConfigurator {
    private restClient;

    constructor(serverUrl: string) {
        this.restClient = new ServerRestClient(serverUrl);
    }

    getSettings = async () => {
        return this.restClient.getSettings();
    };

    getPerformance = async () => {
        return this.restClient.getPerformance();
    };

    updateSettings = async (key: ServerSettingKey, val: string) => {
        return this.restClient.updateSettings(key, val);
    };

    uploadFile2 = async (dir: string, file: File, onprogress: (progress: number, end: boolean) => void) => {
        return this.restClient.uploadFile2(dir, file, onprogress);
    };

    uploadFile = async (buf: ArrayBuffer, filename: string, onprogress: (progress: number, end: boolean) => void) => {
        return this.restClient.uploadFile(buf, filename, onprogress);
    };

    concatUploadedFile = async (filename: string, chunkNum: number) => {
        return this.restClient.concatUploadedFile(filename, chunkNum);
    };

    loadModel = async (slot: number, isHalf: boolean, params: string = "{}") => {
        return this.restClient.loadModel(slot, isHalf, params);
    };

    uploadAssets = async (params: string) => {
        return this.restClient.uploadAssets(params);
    };

    getModelType = async () => {
        return this.restClient.getModelType();
    };

    export2onnx = async () => {
        return this.restClient.export2onnx();
    };

    mergeModel = async (req: MergeModelRequest) => {
        return this.restClient.mergeModel(req);
    };

    updateModelDefault = async () => {
        return this.restClient.updateModelDefault();
    };

    updateModelInfo = async (slot: number, key: string, val: string) => {
        return this.restClient.updateModelInfo(slot, key, val);
    };
}
