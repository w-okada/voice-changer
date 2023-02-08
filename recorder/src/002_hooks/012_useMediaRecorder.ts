import { VoiceFocusDeviceTransformer, VoiceFocusTransformDevice } from "amazon-chime-sdk-js";
import { useEffect, useMemo, useState } from "react";
import { Duplex, DuplexOptions } from "readable-stream";
import MicrophoneStream from "microphone-stream";
import { useAppSetting } from "../003_provider/AppSettingProvider";

export type MediaRecorderState = {
    micMediaStream: MediaStream | undefined,
    vfMediaStream: MediaStream | undefined
}
export type MediaRecorderStateAndMethod = MediaRecorderState & {
    setNewAudioInputDevice: (deviceId: string) => Promise<void>

    startRecord: () => void
    pauseRecord: () => void
    clearRecordedData: () => void
    getRecordedDataBlobs: () => {
        micWavBlob: Blob;
        vfWavBlob: Blob;
        micDuration: number;
        vfDuration: number;
        micSamples: Float32Array;
        vfSamples: Float32Array;
    }
}


// AudioInputデータを蓄積するAudiowStreamer
class AudioStreamer extends Duplex {
    chunks: Float32Array[] = []
    SampleRate: number
    constructor(options: DuplexOptions & {
        SampleRate: number
    }) {
        super(options);
        this.SampleRate = options.SampleRate
    }

    private initializeData = () => {
        this.chunks = []
    }

    clearRecordedData = () => {
        this.initializeData()
    }

    // 蓄積したデータをWavに変換して返す
    getRecordedData = () => {
        const sampleSize = this.chunks.reduce((prev, cur) => {
            return prev + cur.length
        }, 0)
        const samples = new Float32Array(sampleSize);
        let sampleIndex = 0

        for (let i = 0; i < this.chunks.length; i++) {
            for (let j = 0; j < this.chunks[i].length; j++) {
                samples[sampleIndex] = this.chunks[i][j];
                sampleIndex++;
            }
        }

        const writeString = (view: DataView, offset: number, string: string) => {
            for (var i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        const floatTo16BitPCM = (output: DataView, offset: number, input: Float32Array) => {
            for (var i = 0; i < input.length; i++, offset += 2) {
                var s = Math.max(-1, Math.min(1, input[i]));
                output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            }
        };


        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);
        // https://www.youfit.co.jp/archives/1418
        writeString(view, 0, 'RIFF');  // RIFFヘッダ
        view.setUint32(4, 32 + samples.length * 2, true); // これ以降のファイルサイズ
        writeString(view, 8, 'WAVE'); // WAVEヘッダ
        writeString(view, 12, 'fmt '); // fmtチャンク
        view.setUint32(16, 16, true); // fmtチャンクのバイト数
        view.setUint16(20, 1, true); // フォーマットID
        view.setUint16(22, 1, true); // チャンネル数
        view.setUint32(24, this.SampleRate, true); // サンプリングレート
        view.setUint32(28, this.SampleRate * 2, true); // データ速度
        view.setUint16(32, 2, true); // ブロックサイズ
        view.setUint16(34, 16, true); // サンプルあたりのビット数
        writeString(view, 36, 'data'); // dataチャンク
        view.setUint32(40, samples.length * 2, true); // 波形データのバイト数
        floatTo16BitPCM(view, 44, samples); // 波形データ
        console.log(view)
        const audioBlob = new Blob([view], { type: 'audio/wav' });
        const duration = samples.length / this.SampleRate

        return { audioBlob, duration, samples }

    }

    // AudioInputを蓄積
    public _write(chunk: AudioBuffer, _encoding: any, callback: any) {
        const buffer = chunk.getChannelData(0);
        // console.log("SAMPLERATE:", chunk.sampleRate, chunk.numberOfChannels, chunk.length)
        var bufferData = new Float32Array(chunk.length);
        for (var i = 0; i < chunk.length; i++) {
            bufferData[i] = buffer[i];
        }
        this.chunks.push(bufferData)
        callback();
    }
}

export const useMediaRecorder = (): MediaRecorderStateAndMethod => {
    const { applicationSetting, deviceManagerState } = useAppSetting()
    const audioContext = useMemo(() => {
        return new AudioContext({ sampleRate: applicationSetting.applicationSetting.sample_rate });
    }, [])
    const [voiceFocusDeviceTransformer, setVoiceFocusDeviceTransformer] = useState<VoiceFocusDeviceTransformer>();
    const [voiceFocusTransformDevice, setVoiceFocusTransformDevice] = useState<VoiceFocusTransformDevice | null>(null)
    const outputNode = useMemo(() => {
        return audioContext.createMediaStreamDestination();
    }, [])

    const [micMediaStream, setMicMediaStream] = useState<MediaStream>()
    const [vfMediaStream, setVfMediaStream] = useState<MediaStream>()

    // 生の(ノイキャンなしの)データ蓄積用Streamer
    const micAudioStreamer = useMemo(() => {
        return new AudioStreamer({ objectMode: true, SampleRate: applicationSetting.applicationSetting.sample_rate })
    }, [])
    const micStream = useMemo(() => {
        const s = new MicrophoneStream({
            objectMode: true,
            bufferSize: 1024,
            context: audioContext
        });
        s.pipe(micAudioStreamer)
        return s
    }, [])

    // ノイキャンしたデータの蓄積用Streamer
    const vfAudioStreamer = useMemo(() => {
        return new AudioStreamer({ objectMode: true, SampleRate: applicationSetting.applicationSetting.sample_rate })
    }, [])
    const vfStream = useMemo(() => {
        const s = new MicrophoneStream({
            objectMode: true,
            bufferSize: 1024,
            context: audioContext
        })
        s.pipe(vfAudioStreamer)
        return s
    }, [])


    // AudioInput変更のトリガー
    useEffect(() => {
        setNewAudioInputDevice(deviceManagerState.audioInputDeviceId || "")
    }, [deviceManagerState.audioInputDeviceId])

    // AudioInput変更のトリガーによりinputのパイプラインを再生成する
    const setNewAudioInputDevice = async (deviceId: string) => {
        console.log("setNewAudioInputDevice", deviceId)
        let vf = voiceFocusDeviceTransformer
        if (!vf) {
            vf = await VoiceFocusDeviceTransformer.create({ variant: 'c20' })
            setVoiceFocusDeviceTransformer(vf)
        }
        if (micMediaStream) {
            micMediaStream.getTracks().forEach(x => {
                x.stop()
            })
        }

        const constraints: MediaStreamConstraints = {
            audio: {
                deviceId: deviceId,
                sampleRate: applicationSetting.applicationSetting.sample_rate,
                // sampleSize: 16,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
            },
            video: false
        }

        const newMicMediaStream = await navigator.mediaDevices.getUserMedia(constraints)
        // newMicMediaStream.getTracks().forEach(t => {
        //     console.log("Capability:", t.getCapabilities())
        //     console.log("Constraint:", t.getConstraints())
        // })
        let currentDevice = voiceFocusTransformDevice
        if (!currentDevice) {
            currentDevice = (await vf.createTransformDevice(newMicMediaStream)) || null;
            setVoiceFocusTransformDevice(currentDevice)
        } else {
            currentDevice.chooseNewInnerDevice(newMicMediaStream)
        }

        const nodeToVF = audioContext.createMediaStreamSource(newMicMediaStream);

        const voiceFocusNode = await currentDevice!.createAudioNode(audioContext);
        nodeToVF.connect(voiceFocusNode.start);
        voiceFocusNode.end.connect(outputNode);

        setMicMediaStream(newMicMediaStream)
        setVfMediaStream(outputNode.stream)

        micStream.setStream(newMicMediaStream)
        micStream.pauseRecording()
        vfStream.setStream(outputNode.stream)
        vfStream.pauseRecording()
    }


    const startRecord = () => {
        console.log("start record")
        micAudioStreamer.clearRecordedData()
        micStream!.playRecording()
        vfAudioStreamer.clearRecordedData()
        vfStream!.playRecording()
    }

    const pauseRecord = () => {
        micStream!.pauseRecording()
        vfStream!.pauseRecording()
    }
    const clearRecordedData = () => {
        micAudioStreamer.clearRecordedData()
        vfAudioStreamer.clearRecordedData()
    }
    const getRecordedDataBlobs = () => {
        const { audioBlob: micWavBlob, duration: micDuration, samples: micSamples } = micAudioStreamer.getRecordedData()
        const { audioBlob: vfWavBlob, duration: vfDuration, samples: vfSamples } = vfAudioStreamer.getRecordedData()
        return { micWavBlob, vfWavBlob, micDuration, vfDuration, micSamples, vfSamples }
    }

    const retVal: MediaRecorderStateAndMethod = {
        micMediaStream,
        vfMediaStream,
        setNewAudioInputDevice,

        startRecord,
        pauseRecord,
        clearRecordedData,
        getRecordedDataBlobs
    }

    return retVal

}