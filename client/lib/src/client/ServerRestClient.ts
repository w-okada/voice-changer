import { MergeModelRequest, OnnxExporterInfo, ServerInfo, ServerSettingKey } from "../const";

type FileChunk = {
    hash: number;
    chunk: ArrayBuffer;
};

export class ServerRestClient {
    private serverUrl = "";

    constructor(serverUrl: string) {
        this.serverUrl = serverUrl;
    }

    getSettings = async () => {
        const url = this.serverUrl + "/info";
        const info = await new Promise<ServerInfo>((resolve) => {
            const request = new Request(url, {
                method: "GET",
            });
            fetch(request).then(async (response) => {
                const json = (await response.json()) as ServerInfo;
                resolve(json);
            });
        });
        return info;
    };

    getPerformance = async () => {
        const url = this.serverUrl + "/performance";
        const info = await new Promise<number[]>((resolve) => {
            const request = new Request(url, {
                method: "GET",
            });
            fetch(request).then(async (response) => {
                const json = (await response.json()) as number[];
                resolve(json);
            });
        });
        return info;
    };

    updateSettings = async (key: ServerSettingKey, val: string) => {
        const url = this.serverUrl + "/update_settings";
        const info = await new Promise<ServerInfo>(async (resolve) => {
            const formData = new FormData();
            formData.append("key", key);
            formData.append("val", val);
            const request = new Request(url, {
                method: "POST",
                body: formData,
            });
            const res = (await (await fetch(request)).json()) as ServerInfo;
            resolve(res);
        });
        return info;
    };

    uploadFile2 = async (dir: string, file: File, onprogress: (progress: number, end: boolean) => void) =>
        new Promise((resolve, reject) => {
            const url = this.serverUrl + "/upload_file";
            const filename = dir + file.name;
            const xhr = new XMLHttpRequest();
            xhr.upload.addEventListener('progress', e => onprogress((e.loaded / e.total) * 100, false));
            xhr.addEventListener('load', () => resolve({ status: xhr.status, body: xhr.responseText }));
            xhr.addEventListener('error', () => reject(new Error('File upload failed')));
            xhr.addEventListener('abort', () => reject(new Error('File upload aborted')));
            xhr.open('POST', url, true);
            const formData = new FormData();
            formData.append("file", file);
            formData.append("filename", filename);
            xhr.send(formData);
        });

    uploadFile = async (buf: ArrayBuffer, filename: string, onprogress: (progress: number, end: boolean) => void) => {
        const url = this.serverUrl + "/upload_file";
        onprogress(0, false);
        const size = 1024 * 1024;
        const fileChunks: FileChunk[] = [];
        let index = 0; // index値
        for (let cur = 0; cur < buf.byteLength; cur += size) {
            fileChunks.push({
                hash: index++,
                chunk: buf.slice(cur, cur + size),
            });
        }

        const chunkNum = fileChunks.length;
        // console.log("FILE_CHUNKS:", chunkNum, fileChunks)

        while (true) {
            const promises: Promise<void>[] = [];
            for (let i = 0; i < 10; i++) {
                const chunk = fileChunks.shift();
                if (!chunk) {
                    break;
                }
                const p = new Promise<void>((resolve) => {
                    const formData = new FormData();
                    formData.append("file", new Blob([chunk.chunk]));
                    formData.append("filename", `${filename}_${chunk.hash}`);
                    const request = new Request(url, {
                        method: "POST",
                        body: formData,
                    });
                    fetch(request).then(async (_response) => {
                        // console.log(await response.text())
                        resolve();
                    });
                });

                promises.push(p);
            }
            await Promise.all(promises);
            if (fileChunks.length == 0) {
                break;
            }
            onprogress(Math.floor(((chunkNum - fileChunks.length) / (chunkNum + 1)) * 100), false);
        }
        return chunkNum;
    };

    concatUploadedFile = async (filename: string, chunkNum: number) => {
        const url = this.serverUrl + "/concat_uploaded_file";
        await new Promise<void>((resolve) => {
            const formData = new FormData();
            formData.append("filename", filename);
            formData.append("filenameChunkNum", "" + chunkNum);
            const request = new Request(url, {
                method: "POST",
                body: formData,
            });
            fetch(request).then(async (response) => {
                console.log(await response.text());
                resolve();
            });
        });
    };

    loadModel = async (slot: number, isHalf: boolean, params: string = "{}") => {
        if (isHalf == undefined || isHalf == null) {
            console.warn("isHalf is invalid value", isHalf);
            isHalf = false;
        }
        const url = this.serverUrl + "/load_model";
        const info = new Promise<ServerInfo>(async (resolve) => {
            const formData = new FormData();
            formData.append("slot", "" + slot);
            formData.append("isHalf", "" + isHalf);
            formData.append("params", params);

            const request = new Request(url, {
                method: "POST",
                body: formData,
            });
            const res = (await (await fetch(request)).json()) as ServerInfo;
            resolve(res);
        });
        return await info;
    };

    uploadAssets = async (params: string) => {
        const url = this.serverUrl + "/upload_model_assets";
        const info = new Promise<ServerInfo>(async (resolve) => {
            const formData = new FormData();
            formData.append("params", params);

            const request = new Request(url, {
                method: "POST",
                body: formData,
            });
            const res = (await (await fetch(request)).json()) as ServerInfo;
            resolve(res);
        });
        return await info;
    };

    getModelType = async () => {
        const url = this.serverUrl + "/model_type";
        const info = new Promise<ServerInfo>(async (resolve) => {
            const request = new Request(url, {
                method: "GET",
            });
            const res = (await (await fetch(request)).json()) as ServerInfo;
            resolve(res);
        });
        return await info;
    };

    export2onnx = async () => {
        const url = this.serverUrl + "/onnx";
        const info = new Promise<OnnxExporterInfo>(async (resolve) => {
            const request = new Request(url, {
                method: "GET",
            });
            const res = (await (await fetch(request)).json()) as OnnxExporterInfo;
            resolve(res);
        });
        return await info;
    };

    mergeModel = async (req: MergeModelRequest) => {
        const url = this.serverUrl + "/merge_model";
        const info = new Promise<ServerInfo>(async (resolve) => {
            const formData = new FormData();
            formData.append("request", JSON.stringify(req));

            const request = new Request(url, {
                method: "POST",
                body: formData,
            });
            const res = (await (await fetch(request)).json()) as ServerInfo;
            console.log("RESPONSE", res);
            resolve(res);
        });
        return await info;
    };

    updateModelDefault = async () => {
        const url = this.serverUrl + "/update_model_default";
        const info = new Promise<ServerInfo>(async (resolve) => {
            const request = new Request(url, {
                method: "POST",
            });
            const res = (await (await fetch(request)).json()) as ServerInfo;
            console.log("RESPONSE", res);
            resolve(res);
        });
        return await info;
    };

    updateModelInfo = async (slot: number, key: string, val: string) => {
        const url = this.serverUrl + "/update_model_info";
        const newData = { slot, key, val };

        const info = new Promise<ServerInfo>(async (resolve) => {
            const formData = new FormData();
            formData.append("newData", JSON.stringify(newData));

            const request = new Request(url, {
                method: "POST",
                body: formData,
            });
            const res = (await (await fetch(request)).json()) as ServerInfo;
            console.log("RESPONSE", res);
            resolve(res);
        });
        return await info;
    };

    // VoiceChangerWorkletNodeから呼び出される
    //// Restで音声変換
    postVoice = async (timestamp: number, buffer: ArrayBuffer) => {
        const url = this.serverUrl + "/test";

        const obj = {
            timestamp,
            buffer: Buffer.from(buffer).toString("base64"),
        };
        const body = JSON.stringify(obj);

        const res = await fetch(`${url}`, {
            method: "POST",
            headers: {
                Accept: "application/json",
                "Content-Type": "application/json",
            },
            body: body,
        });

        try {
            const receivedJson = await res.json();
            const changedVoiceBase64 = receivedJson["changedVoiceBase64"];
            const buf = Buffer.from(changedVoiceBase64, "base64");
            const ab = new ArrayBuffer(buf.length);
            const view = new Uint8Array(ab);
            for (let i = 0; i < buf.length; ++i) {
                view[i] = buf[i];
            }
            return ab;
        } catch (e) {
            console.log("Exception:", e);
            return new ArrayBuffer(10);
        }
    };
}
