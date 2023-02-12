import { melSpectrogram, powerToDb } from "./100_components/002_parts/207-1MelSpectrogramUtil"

export const AudioOutputElementId = "audio-output-element"

export const generateWavFileName = (prefix: string, index: number) => {
    const indexString = String(index + 1).padStart(3, '0')
    return `${prefix}${indexString}.wav`
}
export const generateTextFileName = (prefix: string, index: number) => {
    const indexString = String(index + 1).padStart(3, '0')
    return `${prefix}${indexString}.txt`
}

export const generateDataNameForLocalStorage = (prefix: string, index: number) => {
    const indexString = String(index + 1).padStart(3, '0')
    const dataName = `${prefix}${indexString}_mic`
    return { dataName }
}
export const generateRegionNameForLocalStorage = (prefix: string, index: number) => {
    const indexString = String(index + 1).padStart(3, '0')
    const regionString = `${prefix}${indexString}_region`
    return regionString
}


const writeString = (view: DataView, offset: number, string: string) => {
    for (var i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
};

export const generateEmptyWav = () => {
    const buffer = new ArrayBuffer(44 + 0 * 2);
    const view = new DataView(buffer);
    // https://www.youfit.co.jp/archives/1418
    writeString(view, 0, 'RIFF');  // RIFFヘッダ
    view.setUint32(4, 32 + 0 * 2, true); // これ以降のファイルサイズ
    writeString(view, 8, 'WAVE'); // WAVEヘッダ
    writeString(view, 12, 'fmt '); // fmtチャンク
    view.setUint32(16, 16, true); // fmtチャンクのバイト数
    view.setUint16(20, 1, true); // フォーマットID
    view.setUint16(22, 1, true); // チャンネル数
    view.setUint32(24, 48000, true); // サンプリングレート
    view.setUint32(28, 48000 * 2, true); // データ速度
    view.setUint16(32, 2, true); // ブロックサイズ
    view.setUint16(34, 16, true); // サンプルあたりのビット数
    writeString(view, 36, 'data'); // dataチャンク
    view.setUint32(40, 0 * 2, true); // 波形データのバイト数
    // floatTo16BitPCM(view, 44, samples); // 波形データ
    // console.log(view)
    const audioBlob = new Blob([view], { type: 'audio/wav' });
    // const duration = samples.length / SampleRate
    return audioBlob

}

export const convert48KhzTo24Khz = async (blob: Blob, startSec?: number, endSec?: number) => {
    blob.arrayBuffer;
    const oldView = new DataView(await blob.arrayBuffer());
    const sampleBytes = oldView.getUint32(40, true); // サンプルデータサイズ = 長さ * 2byte(16bit)
    const sampleLength24Khz = Math.floor(sampleBytes / 2 / 2) + 1;
    // サンプルデータサイズ　 / 2bytes(16bit) => サンプル数(48Khz),
    // サンプル数(48Khz) / 2 = サンプル数(24Khz)　※ 小数点切り捨て + 1

    const startIndex = startSec ? Math.floor(startSec * 24000) : 0;
    const endIndex = endSec ? Math.floor(endSec * 24000) : sampleLength24Khz - 1;
    // console.log("index:::", startIndex, endIndex, startSec, endSec)
    let sampleNum = endIndex - startIndex;
    if (sampleNum > sampleLength24Khz) {
        sampleNum = sampleLength24Khz;
    }

    // console.log("cut...", startIndex, endIndex, sampleNum);

    const buffer = new ArrayBuffer(44 + sampleNum * 2);
    const newView = new DataView(buffer);
    // https://www.youfit.co.jp/archives/1418
    writeString(newView, 0, "RIFF"); // RIFFヘッダ
    newView.setUint32(4, 32 + sampleNum * 2, true); // これ以降のファイルサイズ
    writeString(newView, 8, "WAVE"); // WAVEヘッダ
    writeString(newView, 12, "fmt "); // fmtチャンク
    newView.setUint32(16, 16, true); // fmtチャンクのバイト数
    newView.setUint16(20, 1, true); // フォーマットID
    newView.setUint16(22, 1, true); // チャンネル数
    newView.setUint32(24, 24000, true); // サンプリングレート
    newView.setUint32(28, 24000 * 2, true); // データ速度
    newView.setUint16(32, 2, true); // ブロックサイズ
    newView.setUint16(34, 16, true); // サンプルあたりのビット数
    writeString(newView, 36, "data"); // dataチャンク
    newView.setUint32(40, sampleNum * 2, true); // 波形データのバイト数
    const offset = 44;
    // console.log("converting...", sampleBytes);
    for (let i = 0; i < sampleNum; i++) {
        try {
            const org = oldView.getInt16(offset + 4 * (i + startIndex), true);
            newView.setInt16(offset + 2 * i, org, true);
        } catch (e) {
            console.log(e, "reading...", offset + 4 * i, 4 * i);
            break;
        }
    }
    const audioBlob = new Blob([newView], { type: "audio/wav" });
    return audioBlob;
};



export const drawMel = (data: Float32Array, sampleRate: number) => {
    const canvas = document.createElement("canvas")

    const sp_t = melSpectrogram(data, { sampleRate: sampleRate })
    const sp = powerToDb(sp_t)
    const width = sp.length
    const height = sp[0].length
    canvas.width = width
    canvas.height = height
    const img = new ImageData(width, height)
    for (let i = 0; i < height; i++) {
        for (let j = 0; j < width; j++) {
            const offset = ((i * width) + j) * 4
            const data = sp[j][height - i]
            // console.log(offset)
            img.data[offset + 0] = 0
            img.data[offset + 1] = ((data + 100) / 100) * 255
            img.data[offset + 2] = 0
            img.data[offset + 3] = 255
        }
    }
    const ctx = canvas.getContext("2d")!
    ctx.putImageData(img, 0, 0)
    const png = canvas.toDataURL('image/png')
    return png
}