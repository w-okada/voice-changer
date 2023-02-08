export type DeviceInfo = {
    label: string,
    deviceId: string,
}
export type UpdateListener = {
    update: () => void
}

//////////////////////////////
// Class
//////////////////////////////
export class DeviceManager {
    realAudioInputDevices: DeviceInfo[] = []
    realVideoInputDevices: DeviceInfo[] = []
    realAudioOutputDevices: DeviceInfo[] = []
    updateListener: UpdateListener = {
        update: () => { console.log("update devices") }
    }
    setUpdateListener = (updateListener: UpdateListener) => {
        this.updateListener = updateListener
    }


    // (A) Device List生成
    reloadDevices = async () => {
        try {
            await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        } catch (e) {
            console.warn("Enumerate device error::", e)
        }
        const mediaDeviceInfos = await navigator.mediaDevices.enumerateDevices();

        this.realAudioInputDevices = mediaDeviceInfos.filter(x => { return x.kind === "audioinput" }).map(x => { return { label: x.label, deviceId: x.deviceId } })
        this.realVideoInputDevices = mediaDeviceInfos.filter(x => { return x.kind === "videoinput" }).map(x => { return { label: x.label, deviceId: x.deviceId } })
        this.realAudioOutputDevices = mediaDeviceInfos.filter(x => { return x.kind === "audiooutput" }).map(x => { return { label: x.label, deviceId: x.deviceId } })

        this.realAudioInputDevices.push({ label: "none", deviceId: "none" })
        this.realVideoInputDevices.push({ label: "none", deviceId: "none" })


        this.updateListener.update()
    }
}
