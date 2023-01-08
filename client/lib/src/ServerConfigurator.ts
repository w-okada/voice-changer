import { ServerInfo, ServerSettingKey } from "./const";


type FileChunk = {
    hash: number,
    chunk: Blob
}
export class ServerConfigurator {
    private serverUrl = ""
    getSettings = async () => {
        const url = this.serverUrl + "/info"
        const info = await new Promise<ServerInfo>((resolve) => {
            const request = new Request(url, {
                method: 'GET',
            });
            fetch(request).then(async (response) => {
                const json = await response.json() as ServerInfo
                resolve(json)
            })
        })
        return info
    }

    setServerUrl = (serverUrl: string) => {
        this.serverUrl = serverUrl
        console.log(`[ServerConfigurator] Server URL: ${this.serverUrl}`)
    }

    updateSettings = async (key: ServerSettingKey, val: string) => {
        const url = this.serverUrl + "/update_setteings"
        const p = new Promise<void>((resolve) => {
            const formData = new FormData();
            formData.append("key", key);
            formData.append("val", val);
            const request = new Request(url, {
                method: 'POST',
                body: formData,
            });
            fetch(request).then(async (response) => {
                console.log(await response.json())
                resolve()
            })
        })
        const info = await p
        return info
    }

    uploadFile = async (file: File, onprogress: (progress: number, end: boolean) => void) => {
        const url = this.serverUrl + "/upload_file"
        onprogress(0, false)
        const size = 1024 * 1024;
        const fileChunks: FileChunk[] = [];
        let index = 0; // index値
        for (let cur = 0; cur < file.size; cur += size) {
            fileChunks.push({
                hash: index++,
                chunk: file.slice(cur, cur + size),
            });
        }

        const chunkNum = fileChunks.length
        console.log("FILE_CHUNKS:", chunkNum, fileChunks)


        while (true) {
            const promises: Promise<void>[] = []
            for (let i = 0; i < 10; i++) {
                const chunk = fileChunks.shift()
                if (!chunk) {
                    break
                }
                const p = new Promise<void>((resolve) => {
                    const formData = new FormData();
                    formData.append("file", chunk.chunk);
                    formData.append("filename", `${file.name}_${chunk.hash}`);
                    const request = new Request(url, {
                        method: 'POST',
                        body: formData,
                    });
                    fetch(request).then(async (response) => {
                        console.log(await response.text())
                        resolve()
                    })
                })

                promises.push(p)
            }
            await Promise.all(promises)
            if (fileChunks.length == 0) {
                break
            }
            onprogress(Math.floor(((chunkNum - fileChunks.length) / (chunkNum + 1)) * 100), false)
        }
        return chunkNum
    }

    concatUploadedFile = async (file: File, chunkNum: number) => {
        const url = this.serverUrl + "/concat_uploaded_file"
        new Promise<void>((resolve) => {
            const formData = new FormData();
            formData.append("filename", file.name);
            formData.append("filenameChunkNum", "" + chunkNum);
            const request = new Request(url, {
                method: 'POST',
                body: formData,
            });
            fetch(request).then(async (response) => {
                console.log(await response.text())
                resolve()
            })
        })
    }

    loadModel = async (configFile: File, pyTorchModelFile: File | null, onnxModelFile: File | null) => {
        const url = this.serverUrl + "/load_model"
        const loadP = new Promise<void>((resolve) => {
            const formData = new FormData();
            formData.append("pyTorchModelFilename", pyTorchModelFile?.name || "-");
            formData.append("onnxModelFilename", onnxModelFile?.name || "-");
            formData.append("configFilename", configFile.name);
            const request = new Request(url, {
                method: 'POST',
                body: formData,
            });
            fetch(request).then(async (response) => {
                console.log(await response.text())
                resolve()
            })
        })
        await loadP
    }
}