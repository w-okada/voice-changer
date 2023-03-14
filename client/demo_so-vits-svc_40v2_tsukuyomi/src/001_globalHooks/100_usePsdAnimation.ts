import { generateConfig, PSDAnimatorParams, WorkerManager } from "@dannadori/psdanimator"
import { useMemo } from "react"
import { TSUKUYOMI_CANVAS } from "../const"


type PsdAnimationState = {
};

export type PsdAnimationStateAndMethod = PsdAnimationState & {
    loadPsd: () => Promise<void>
    setMotion: () => Promise<void>
    start: () => Promise<void>
    switchNormalMotion: () => Promise<void>
    switchTalkingMotion: () => Promise<void>
}

export const usePsdAnimation = (): PsdAnimationStateAndMethod => {
    const w = useMemo(() => {
        return new WorkerManager()
    }, [])

    const loadPsd = async () => {
        const psdFile = await (await fetch("./assets/tsukuyomi.psd")).arrayBuffer()
        const canvas = document.getElementById(TSUKUYOMI_CANVAS) as HTMLCanvasElement
        // const c = generateConfig(psdFile, canvas, 640, 480, true)
        const c = generateConfig(psdFile, canvas, 640, 480, false)
        c.transfer = [c.canvas]
        await w.init(c)
        console.log("[psd animator] Initialized")
    }


    // "ROOT_!眉　_*通常眉",
    // "ROOT_!眉　_*通常眉：少し下げ",
    // "ROOT_!眉　_*喜び・驚き",
    // "ROOT_!目_*通常目",
    // "ROOT_!目_*目を少し細める ★デフォルト",
    // "ROOT_!目_*目を細める",
    // "ROOT_!目_*目を閉じる",
    // "ROOT_!目_*目を閉じて微笑む",
    // "ROOT_!頬　_*通常",
    // "ROOT_!口 _*口を閉じて微笑む",
    // "ROOT_!口 _*小さく笑う",
    // "ROOT_!口 _*大きく笑う",
    // "ROOT_!髪（おさげ）_!髪（おさげ）　※選択肢なし",
    // "ROOT_!左腕_*左手を下げる",
    // "ROOT_!左腕_*左手でご案内",
    // "ROOT_!左腕_*左手を胸の前に：グー",
    // "ROOT_!左腕_*左手を胸の前に：パー",
    // "ROOT_!左腕_*左手を口に当てる",
    // "ROOT_!右腕_*右手を下げる",
    // "ROOT_!右腕_*右手でご案内",
    // "ROOT_!右腕_*右手を胸の前に：グー",
    // "ROOT_!右腕_*右手を胸の前に：パー",
    // "ROOT_!右腕_*右手を口に当てる",
    // "ROOT_!体_!体　※選択肢なし"

    const setMotion = async () => {
        const p1: PSDAnimatorParams = {
            type: "SET_MOTION",
            motion: [
                { "mode": "normal", "z_index": 0, "number": 10, "layer_path": "ROOT_!体_!体　※選択肢なし" },
                { "mode": "normal", "z_index": 1, "number": 200, "layer_path": "ROOT_!右腕_*右手を下げる" },
                { "mode": "normal", "z_index": 1, "number": 40, "layer_path": "ROOT_!右腕_*右手を口に当てる" },
                { "mode": "normal", "z_index": 2, "number": 10, "layer_path": "ROOT_!左腕_*左手を下げる" },
                { "mode": "normal", "z_index": 3, "number": 10, "layer_path": "ROOT_!髪（おさげ）_!髪（おさげ）　※選択肢なし" },
                { "mode": "normal", "z_index": 4, "number": 10, "layer_path": "ROOT_!口 _*口を閉じて微笑む" },
                { "mode": "normal", "z_index": 5, "number": 10, "layer_path": "ROOT_!頬　_*通常" },
                { "mode": "normal", "z_index": 6, "number": 10, "layer_path": "ROOT_!目_*目を少し細める ★デフォルト" },
                { "mode": "normal", "z_index": 7, "number": 100, "layer_path": "ROOT_!眉　_*通常眉" },
                { "mode": "normal", "z_index": 7, "number": 10, "layer_path": "ROOT_!眉　_*通常眉：少し下げ" },


                { "mode": "talking", "z_index": 0, "number": 10, "layer_path": "ROOT_!体_!体　※選択肢なし" },
                { "mode": "talking", "z_index": 1, "number": 200, "layer_path": "ROOT_!右腕_*右手を下げる" },
                { "mode": "talking", "z_index": 1, "number": 40, "layer_path": "ROOT_!右腕_*右手を口に当てる" },
                { "mode": "talking", "z_index": 2, "number": 10, "layer_path": "ROOT_!左腕_*左手を下げる" },
                { "mode": "talking", "z_index": 3, "number": 10, "layer_path": "ROOT_!髪（おさげ）_!髪（おさげ）　※選択肢なし" },
                { "mode": "talking", "z_index": 4, "number": 3, "layer_path": "ROOT_!口 _*口を閉じて微笑む" },
                { "mode": "talking", "z_index": 4, "number": 3, "layer_path": "ROOT_!口 _*小さく笑う" },
                { "mode": "talking", "z_index": 4, "number": 3, "layer_path": "ROOT_!口 _*大きく笑う" },
                { "mode": "talking", "z_index": 5, "number": 10, "layer_path": "ROOT_!頬　_*通常" },
                { "mode": "talking", "z_index": 6, "number": 10, "layer_path": "ROOT_!目_*目を少し細める ★デフォルト" },
                { "mode": "talking", "z_index": 7, "number": 100, "layer_path": "ROOT_!眉　_*通常眉" },
                { "mode": "talking", "z_index": 7, "number": 10, "layer_path": "ROOT_!眉　_*通常眉：少し下げ" },

            ],
            transfer: []
        }
        await w.execute(p1)
    }


    const start = async () => {
        const p3: PSDAnimatorParams = {
            type: "START",
            transfer: []
        }
        await w.execute(p3)
    }
    const switchNormalMotion = async () => {
        const p2: PSDAnimatorParams = {
            type: "SWITCH_MOTION_MODE",
            motionMode: "normal",
            transfer: []
        }
        await w.execute(p2)
    }
    const switchTalkingMotion = async () => {
        const p2: PSDAnimatorParams = {
            type: "SWITCH_MOTION_MODE",
            motionMode: "talking",
            transfer: []
        }
        await w.execute(p2)
    }
    return {
        loadPsd,
        setMotion,
        switchNormalMotion,
        switchTalkingMotion,
        start
    }
}