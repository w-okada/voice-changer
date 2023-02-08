// Original is from https://github.com/magenta/magenta-js (d8a7668 on 2 Nov 2021 Git stats)
// I extracted functions from original repos.to use.
// @ts-ignore
import * as FFT from 'fft.js';

type SpecParams = {
    sampleRate: number;
    hopLength?: number;
    winLength?: number;
    nFft?: number;
    nMels?: number;
    power?: number;
    fMin?: number;
    fMax?: number;
}

const hannWindow = (length: number) => {
    const win = new Float32Array(length);
    for (let i = 0; i < length; i++) {
        win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (length - 1)));
    }
    return win;
}

const padConstant = (data: Float32Array, padding: number | number[]) => {
    let padLeft, padRight;
    if (typeof padding === 'object') {
        [padLeft, padRight] = padding;
    } else {
        padLeft = padRight = padding;
    }
    const out = new Float32Array(data.length + padLeft + padRight);
    out.set(data, padLeft);
    return out;
}
const padCenterToLength = (data: Float32Array, length: number) => {
    // If data is longer than length, error!
    if (data.length > length) {
        throw new Error('Data is longer than length.');
    }

    const paddingLeft = Math.floor((length - data.length) / 2);
    const paddingRight = length - data.length - paddingLeft;
    return padConstant(data, [paddingLeft, paddingRight]);
}

const padReflect = (data: Float32Array, padding: number) => {
    const out = padConstant(data, padding);
    for (let i = 0; i < padding; i++) {
        // Pad the beginning with reflected values.
        out[i] = out[2 * padding - i];
        // Pad the end with reflected values.
        out[out.length - i - 1] = out[out.length - 2 * padding + i - 1];
    }
    return out;
}

const frame = (
    data: Float32Array, frameLength: number,
    hopLength: number): Float32Array[] => {
    const bufferCount = Math.floor((data.length - frameLength) / hopLength) + 1;
    const buffers = Array.from(
        { length: bufferCount }, (_x, _i) => new Float32Array(frameLength));
    for (let i = 0; i < bufferCount; i++) {
        const ind = i * hopLength;
        const buffer = data.slice(ind, ind + frameLength);
        buffers[i].set(buffer);
        // In the end, we will likely have an incomplete buffer, which we should
        // just ignore.
        if (buffer.length !== frameLength) {
            continue;
        }
    }
    return buffers;
}


const applyWindow = (buffer: Float32Array, win: Float32Array) => {
    if (buffer.length !== win.length) {
        console.error(
            `Buffer length ${buffer.length} != window length ${win.length}.`);
        return null;
    }

    const out = new Float32Array(buffer.length);
    for (let i = 0; i < buffer.length; i++) {
        out[i] = win[i] * buffer[i];
    }
    return out;
}


const fft = (y: Float32Array) => {
    const fft = new FFT(y.length);
    const out = fft.createComplexArray();
    const data = fft.toComplexArray(y, null);
    fft.transform(out, data);
    return out;
}
const stft = (y: Float32Array, params: SpecParams): Float32Array[] => {
    const nFft = params.nFft || 2048;
    const winLength = params.winLength || nFft;
    const hopLength = params.hopLength || Math.floor(winLength / 4);

    let fftWindow = hannWindow(winLength);

    // Pad the window to be the size of nFft.
    fftWindow = padCenterToLength(fftWindow, nFft);

    // Pad the time series so that the frames are centered.
    y = padReflect(y, Math.floor(nFft / 2));

    // Window the time series.
    const yFrames = frame(y, nFft, hopLength);
    // Pre-allocate the STFT matrix.
    const stftMatrix: Float32Array[] = [];

    const width = yFrames.length;
    const height = nFft + 2;
    for (let i = 0; i < width; i++) {
        // Each column is a Float32Array of size height.
        const col = new Float32Array(height);
        stftMatrix[i] = col;
    }

    for (let i = 0; i < width; i++) {
        // Populate the STFT matrix.
        const winBuffer = applyWindow(yFrames[i], fftWindow);
        const col = fft(winBuffer!);
        stftMatrix[i].set(col.slice(0, height));
    }

    return stftMatrix;
}


const pow = (arr: Float32Array, power: number) => {
    return arr.map((v) => Math.pow(v, power));
}

function magSpectrogram(
    stft: Float32Array[], power: number): [Float32Array[], number] {
    const spec = stft.map((fft) => pow(mag(fft), power));
    const nFft = stft[0].length - 1;
    return [spec, nFft];
}
const mag = (y: Float32Array) => {
    const out = new Float32Array(y.length / 2);
    for (let i = 0; i < y.length / 2; i++) {
        out[i] = Math.sqrt(y[i * 2] * y[i * 2] + y[i * 2 + 1] * y[i * 2 + 1]);
    }
    return out;
}
interface MelParams {
    sampleRate: number;
    nFft?: number;
    nMels?: number;
    fMin?: number;
    fMax?: number;
}
const linearSpace = (start: number, end: number, count: number) => {
    // Include start and endpoints.
    const delta = (end - start) / (count - 1);
    const out = new Float32Array(count);
    for (let i = 0; i < count; i++) {
        out[i] = start + delta * i;
    }
    return out;
}
const calculateFftFreqs = (sampleRate: number, nFft: number) => {
    return linearSpace(0, sampleRate / 2, Math.floor(1 + nFft / 2));
}
const hzToMel = (hz: number): number => {
    return 1125.0 * Math.log(1 + hz / 700.0);
}
function melToHz(mel: number): number {
    return 700.0 * (Math.exp(mel / 1125.0) - 1);
}
const calculateMelFreqs = (
    nMels: number, fMin: number, fMax: number): Float32Array => {
    const melMin = hzToMel(fMin);
    const melMax = hzToMel(fMax);

    // Construct linearly spaced array of nMel intervals, between melMin and
    // melMax.
    const mels = linearSpace(melMin, melMax, nMels);
    const hzs = mels.map((mel) => melToHz(mel));
    return hzs;
}
const internalDiff = (arr: Float32Array): Float32Array => {
    const out = new Float32Array(arr.length - 1);
    for (let i = 0; i < arr.length; i++) {
        out[i] = arr[i + 1] - arr[i];
    }
    return out;
}

const outerSubtract = (arr: Float32Array, arr2: Float32Array): Float32Array[] => {
    const out: Float32Array[] = [];
    for (let i = 0; i < arr.length; i++) {
        out[i] = new Float32Array(arr2.length);
    }
    for (let i = 0; i < arr.length; i++) {
        for (let j = 0; j < arr2.length; j++) {
            out[i][j] = arr[i] - arr2[j];
        }
    }
    return out;
}
const createMelFilterbank = (params: MelParams): Float32Array[] => {
    const fMin = params.fMin || 0;
    const fMax = params.fMax || params.sampleRate / 2;
    const nMels = params.nMels || 128;
    const nFft = params.nFft || 2048;

    // Center freqs of each FFT band.
    const fftFreqs = calculateFftFreqs(params.sampleRate, nFft);
    // (Pseudo) center freqs of each Mel band.
    const melFreqs = calculateMelFreqs(nMels + 2, fMin, fMax);

    const melDiff = internalDiff(melFreqs);
    const ramps = outerSubtract(melFreqs, fftFreqs);
    const filterSize = ramps[0].length;

    const weights: Float32Array[] = [];
    for (let i = 0; i < nMels; i++) {
        weights[i] = new Float32Array(filterSize);
        for (let j = 0; j < ramps[i].length; j++) {
            const lower = -ramps[i][j] / melDiff[i];
            const upper = ramps[i + 2][j] / melDiff[i + 1];
            const weight = Math.max(0, Math.min(lower, upper));
            weights[i][j] = weight;
        }
    }

    // Slaney-style mel is scaled to be approx constant energy per channel.
    for (let i = 0; i < weights.length; i++) {
        // How much energy per channel.
        const enorm = 2.0 / (melFreqs[2 + i] - melFreqs[i]);
        // Normalize by that amount.
        weights[i] = weights[i].map((val) => val * enorm);
    }

    return weights;
}


const applyFilterbank = (
    mags: Float32Array, filterbank: Float32Array[]): Float32Array => {
    if (mags.length !== filterbank[0].length) {
        throw new Error(
            `Each entry in filterbank should have dimensions ` +
            `matching FFT. |mags| = ${mags.length}, ` +
            `|filterbank[0]| = ${filterbank[0].length}.`);
    }

    // Apply each filter to the whole FFT signal to get one value.
    const out = new Float32Array(filterbank.length);
    for (let i = 0; i < filterbank.length; i++) {
        // To calculate filterbank energies we multiply each filterbank with the
        // power spectrum.
        const win = applyWindow(mags, filterbank[i]);
        // Then add up the coefficents.
        out[i] = win!.reduce((a, b) => a + b);
    }
    return out;
}
const applyWholeFilterbank = (
    spec: Float32Array[], filterbank: Float32Array[]): Float32Array[] => {
    // Apply a point-wise dot product between the array of arrays.
    const out: Float32Array[] = [];
    for (let i = 0; i < spec.length; i++) {
        out[i] = applyFilterbank(spec[i], filterbank);
    }
    return out;
}

export const melSpectrogram = (y: Float32Array, params: SpecParams): Float32Array[] => {
    if (!params.power) {
        params.power = 2.0;
    }
    const stftMatrix = stft(y, params);
    const [spec, nFft] = magSpectrogram(stftMatrix, params.power);

    params.nFft = nFft;
    const melBasis = createMelFilterbank(params);
    return applyWholeFilterbank(spec, melBasis);

}

const max = (arr: Float32Array) => {
    return arr.reduce((a, b) => Math.max(a, b));
}

export const powerToDb = (spec: Float32Array[], amin = 1e-10, topDb = 80.0) => {
    const width = spec.length;
    const height = spec[0].length;
    const logSpec: Float32Array[] = [];
    for (let i = 0; i < width; i++) {
        logSpec[i] = new Float32Array(height);
    }
    for (let i = 0; i < width; i++) {
        for (let j = 0; j < height; j++) {
            const val = spec[i][j];
            logSpec[i][j] = 10.0 * Math.log10(Math.max(amin, val));
        }
    }
    if (topDb) {
        if (topDb < 0) {
            throw new Error(`topDb must be non-negative.`);
        }
        for (let i = 0; i < width; i++) {
            const maxVal = max(logSpec[i]);
            for (let j = 0; j < height; j++) {
                logSpec[i][j] = Math.max(logSpec[i][j], maxVal - topDb);
            }
        }
    }
    return logSpec;
}
