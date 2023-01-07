import { OnnxExecutionProvider } from "./const"

const DEBUG = false
const DEBUG_BASE_URL = "http://localhost:18888"

type FileChunk = {
    hash: number,
    chunk: Blob
}

export type ServerInfo = {
    pyTorchModelFile: string,
    onnxModelFile: string,
    configFile: string,
    providers: string[]
}


export const getInfo = async (baseUrl: string) => {
    const getInfoURL = DEBUG ? `${DEBUG_BASE_URL}/info` : `${baseUrl}/info`

    const info = await new Promise<ServerInfo>((resolve) => {
        const request = new Request(getInfoURL, {
            method: 'GET',
        });
        fetch(request).then(async (response) => {
            const json = await response.json() as ServerInfo
            resolve(json)
        })
    })
    return info
}


export const uploadLargeFile = async (baseUrl: string, file: File, onprogress: (progress: number, end: boolean) => void) => {
    const uploadURL = DEBUG ? `${DEBUG_BASE_URL}/upload_file` : `${baseUrl}/upload_file`
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
                const request = new Request(uploadURL, {
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

export const concatUploadedFile = async (baseUrl: string, file: File, chunkNum: number) => {
    const loadModelURL = DEBUG ? `${DEBUG_BASE_URL}/concat_uploaded_file` : `${baseUrl}/concat_uploaded_file`

    new Promise<void>((resolve) => {
        const formData = new FormData();
        formData.append("filename", file.name);
        formData.append("filenameChunkNum", "" + chunkNum);
        const request = new Request(loadModelURL, {
            method: 'POST',
            body: formData,
        });
        fetch(request).then(async (response) => {
            console.log(await response.text())
            resolve()
        })
    })
}

export const loadModel = async (baseUrl: string, configFile: File, pyTorchModelFile: File | null, onnxModelFile: File | null) => {
    const loadModelURL = DEBUG ? `${DEBUG_BASE_URL}/load_model` : `${baseUrl}/load_model`
    const loadP = new Promise<void>((resolve) => {
        const formData = new FormData();
        formData.append("pyTorchModelFilename", pyTorchModelFile?.name || "-");
        formData.append("onnxModelFilename", onnxModelFile?.name || "-");
        formData.append("configFilename", configFile.name);
        const request = new Request(loadModelURL, {
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

export const setOnnxExecutionProvider = async (baseUrl: string, provider: OnnxExecutionProvider) => {
    const url = DEBUG ? `${DEBUG_BASE_URL}/set_onnx_provider` : `${baseUrl}/set_onnx_provider`
    const loadP = new Promise<void>((resolve) => {
        const formData = new FormData();
        formData.append("provider", provider);
        const request = new Request(url, {
            method: 'POST',
            body: formData,
        });
        fetch(request).then(async (response) => {
            console.log(await response.json())
            resolve()
        })
    })
    await loadP
}

// export const uploadModelProps = async (baseUrl: string, modelFile: File, configFile: File, onprogress: (progress: number, end: boolean) => void) => {
//     const uploadURL = DEBUG ? `${DEBUG_BASE_URL}/upload_file` : `${baseUrl}/upload_file`
//     const loadModelURL = DEBUG ? `${DEBUG_BASE_URL}/load_model` : `${baseUrl}/load_model`
//     onprogress(0, false)

//     const chunkNum = await uploadLargeFile(baseUrl, modelFile, (progress: number, _end: boolean) => {
//         onprogress(progress, false)
//     })
//     console.log("model uploaded")


//     const configP = new Promise<void>((resolve) => {
//         const formData = new FormData();
//         formData.append("file", configFile);
//         formData.append("filename", configFile.name);
//         const request = new Request(uploadURL, {
//             method: 'POST',
//             body: formData,
//         });
//         fetch(request).then(async (response) => {
//             console.log(await response.text())
//             resolve()
//         })
//     })

//     await configP
//     console.log("config uploaded")

//     const loadP = new Promise<void>((resolve) => {
//         const formData = new FormData();
//         formData.append("modelFilename", modelFile.name);
//         formData.append("modelFilenameChunkNum", "" + chunkNum);
//         formData.append("configFilename", configFile.name);
//         const request = new Request(loadModelURL, {
//             method: 'POST',
//             body: formData,
//         });
//         fetch(request).then(async (response) => {
//             console.log(await response.text())
//             resolve()
//         })
//     })
//     await loadP
//     onprogress(100, true)
//     console.log("model loaded")
// }


