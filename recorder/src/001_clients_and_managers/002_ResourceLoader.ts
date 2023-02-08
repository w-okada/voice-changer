import { FixedUserData } from "../002_hooks/013_useAudioControllerState";

export const fetchTextResource = async (url: string): Promise<string> => {
    const res = await fetch(url, {
        method: "GET"
    });
    const text = res.text()
    return text;
}

export const postVoice = async (title: string, prefix: string, index: number, blob: Blob) => {
    // const url = `./api/voice/${title}/${prefix}/${index}`
    // const url = `./api/voice`
    // !!!!!!!!!!! COLABのプロキシがRoot直下のパスしか通さない??? !!!!!!
    // !!!!!!!!!!! Bodyで参照、設定コマンドを代替する。 !!!!!!
    const url = `/api`

    const blobBuffer = await blob.arrayBuffer()
    const obj = {
        command: "POST_VOICE",
        data: Buffer.from(blobBuffer).toString("base64"),
        title: title,
        prefix: prefix,
        index: index
    };
    const body = JSON.stringify(obj);

    const res = await fetch(`${url}`, {
        method: "POST",
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: body
    })

    const receivedJson = await res.json()
    const message = receivedJson["message"]
    console.debug("POST VOICE RES:", message)
    return
}

export const getVoice = async (title: string, prefix: string, index: number) => {
    if (!title || !prefix) {
        return null
    }

    const url = `/api`
    const obj = {
        command: "GET_VOICE",
        title: title,
        prefix: prefix,
        index: index
    };
    const body = JSON.stringify(obj);

    const res = await fetch(`${url}`, {
        method: "POST",
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: body
    })


    const receivedJson = await res.json()
    // const message = receivedJson["message"]
    const dataBase64 = receivedJson["data"]
    // console.log("GET VOICE RES:", message, dataBase64)
    if (!dataBase64) {
        return null;
    }
    const buf = Buffer.from(dataBase64, "base64")
    const blob = new Blob([buf.buffer])
    return blob
}

export const postVoice__ = async (title: string, index: number, userData: FixedUserData) => {
    const url = `/api/voice/${title}/${index}`
    const micWavBlob = await userData.micWavBlob!.arrayBuffer()
    const vfWavBlob = await userData.vfWavBlob!.arrayBuffer()
    const micF32 = await userData.micWavSamples!
    const vfF32 = await userData.vfWavSamples!

    const obj = {
        micWavBlob: Buffer.from(micWavBlob).toString("base64"),
        vfWavBlob: Buffer.from(vfWavBlob).toString("base64"),
        micF32: Buffer.from(micF32).toString("base64"),
        vfF32: Buffer.from(vfF32).toString("base64")
    };
    const body = JSON.stringify(obj);

    const res = await fetch(`${url}`, {
        method: "POST",
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: body
    })

    const receivedJson = await res.json()
    const changedVoiceBase64 = receivedJson["changedVoiceBase64"]
    const buf = Buffer.from(changedVoiceBase64, "base64")
    const ab = new ArrayBuffer(buf.length);
    const view = new Uint8Array(ab);
    for (let i = 0; i < buf.length; ++i) {
        view[i] = buf[i];
    }
    return ab
}

