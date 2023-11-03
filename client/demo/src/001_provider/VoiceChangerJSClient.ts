import { create, ConverterType } from "@alexanderolsen/libsamplerate-js";
import { BlockingQueue } from "./_BlockingQueue";
import { WorkerManager, generateConfig, VoiceChangerProcessorInitializeParams, VoiceChangerProcessorConvertParams, FunctionType, VoiceChangerProcessorResult } from "@dannadori/voice-changer-js";

export class VoiceChangerJSClient {
    private wm = new WorkerManager();
    private audioBuffer: Float32Array = new Float32Array(0);
    private audioInputLength = 24000;

    private inputSamplingRate = 48000;
    private outputSamplingRate = 48000;
    private modelInputSamplingRate = 16000;
    private modelOutputSamplingRate = 40000;
    private sem = new BlockingQueue<number>();
    private crossfadeChunks = 1;
    private solaChunks = 0.5;
    constructor() {
        this.sem.enqueue(0);
    }
    private lock = async () => {
        const num = await this.sem.dequeue();
        return num;
    };
    private unlock = (num: number) => {
        this.sem.enqueue(num + 1);
    };

    initialize = async () => {
        console.log("Voice Changer Initializing,,,");
        const baseUrl = "http://127.0.0.1:18888";

        this.wm = new WorkerManager();
        const config = generateConfig();
        config.processorURL = `${baseUrl}/process.js`;
        config.onnxWasmPaths = `${baseUrl}/`;
        await this.wm.init(config);

        const initializeParams: VoiceChangerProcessorInitializeParams = {
            type: FunctionType.initialize,
            inputLength: 24000,
            f0_min: 50,
            f0_max: 1100,
            embPitchUrl: "http://127.0.0.1:18888/models/emb_pit_24000.bin",
            rvcv2InputLength: 148,
            // rvcv2Url: "http://127.0.0.1:18888/models/rvc2v_24000.bin",
            rvcv2Url: "http://127.0.0.1:18888/models/rvc2vnof0_24000.bin",
            transfer: [],
        };

        const res = (await this.wm.execute(initializeParams)) as VoiceChangerProcessorResult;
        console.log("Voice Changer Initialized..", res);
    };

    convert = async (audio: Float32Array): Promise<Float32Array> => {
        console.log("convert start....", audio);
        const lockNum = await this.lock();
        //resample
        const audio_16k = await this.resample(audio, this.inputSamplingRate, this.modelInputSamplingRate);
        //store data and get target data
        //// store
        const newAudioBuffer = new Float32Array(this.audioBuffer.length + audio_16k.length);
        newAudioBuffer.set(this.audioBuffer);
        newAudioBuffer.set(audio_16k, this.audioBuffer.length);
        this.audioBuffer = newAudioBuffer;

        //// Buffering.....
        if (this.audioBuffer.length < this.audioInputLength * 1) {
            console.log(`skip covert length:${this.audioBuffer.length}, audio_16k:${audio_16k.length}`);
            await this.unlock(lockNum);
            return new Float32Array(1);
        } else {
            console.log(`--------------- convert start... length:${this.audioBuffer.length}, audio_16k:${audio_16k.length}`);
        }

        //// get chunks
        let chunkIndex = 0;
        const audioChunks: Float32Array[] = [];
        while (true) {
            const chunkOffset = chunkIndex * this.audioInputLength - (this.crossfadeChunks + this.solaChunks) * 320 * chunkIndex;
            const chunkEnd = chunkOffset + this.audioInputLength;
            if (chunkEnd > this.audioBuffer.length) {
                this.audioBuffer = this.audioBuffer.slice(chunkOffset);
                break;
            } else {
                const chunk = this.audioBuffer.slice(chunkOffset, chunkEnd);
                audioChunks.push(chunk);
            }
            chunkIndex++;
        }

        if (audioChunks.length == 0) {
            await this.unlock(lockNum);
            console.log(`skip covert length:${this.audioBuffer.length}, audio_16k:${audio_16k.length}`);
            return new Float32Array(1);
        }

        //convert (each)
        const convetedAudioChunks: Float32Array[] = [];
        for (let i = 0; i < audioChunks.length; i++) {
            const convertParams: VoiceChangerProcessorConvertParams = {
                type: FunctionType.convert,
                transfer: [audioChunks[i].buffer],
            };
            const res = (await this.wm.execute(convertParams)) as VoiceChangerProcessorResult;
            const converted = new Float32Array(res.transfer[0] as ArrayBuffer);
            console.log(`converted.length:::${i}:${converted.length}`);

            convetedAudioChunks.push(converted);
        }

        //concat
        let totalLength = convetedAudioChunks.reduce((prev, cur) => prev + cur.length, 0);
        let convetedAudio = new Float32Array(totalLength);
        let offset = 0;
        for (let chunk of convetedAudioChunks) {
            convetedAudio.set(chunk, offset);
            offset += chunk.length;
        }
        console.log(`converted.length:::convetedAudio:${convetedAudio.length}`);

        //resample
        // const response = await this.resample(convetedAudio, this.params.modelOutputSamplingRate, this.params.outputSamplingRate);

        const outputDuration = (this.audioInputLength * audioChunks.length - this.crossfadeChunks * 320) / 16000;
        const outputSamples = outputDuration * this.outputSamplingRate;
        const convertedOutputRatio = outputSamples / convetedAudio.length;
        const realOutputSamplingRate = this.modelOutputSamplingRate * convertedOutputRatio;
        console.log(`realOutputSamplingRate:${realOutputSamplingRate}, `, this.modelOutputSamplingRate, convertedOutputRatio);

        // const response2 = await this.resample(convetedAudio, this.params.modelOutputSamplingRate, realOutputSamplingRate);
        const response2 = await this.resample(convetedAudio, this.modelOutputSamplingRate, this.outputSamplingRate);

        console.log(`converted from :${audioChunks.length * this.audioInputLength} to:${convetedAudio.length} to:${response2.length}`);
        console.log(`outputDuration :${outputDuration} outputSamples:${outputSamples}, convertedOutputRatio:${convertedOutputRatio}, realOutputSamplingRate:${realOutputSamplingRate}`);
        await this.unlock(lockNum);
        return response2;
    };

    // Utility
    resample = async (data: Float32Array, srcSampleRate: number, dstSampleRate: number) => {
        const converterType = ConverterType.SRC_SINC_BEST_QUALITY;
        const nChannels = 1;
        const converter = await create(nChannels, srcSampleRate, dstSampleRate, {
            converterType: converterType, // default SRC_SINC_FASTEST. see API for more
        });
        const res = converter.simple(data);
        return res;
    };
}
