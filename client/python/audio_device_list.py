import pyaudio

if __name__ == '__main__':
    audio = pyaudio.PyAudio()
    audio_input_devices = []
    audio_output_devices = []
    audio_devices = {}
    host_apis = []

    for api_index in range(audio.get_host_api_count()):
        host_apis.append(audio.get_host_api_info_by_index(api_index)['name'])

    for x in range(0, audio.get_device_count()):
        device = audio.get_device_info_by_index(x)
        try:
            deviceName = device['name'].encode('shift-jis').decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            deviceName = device['name']

        deviceIndex = device['index']
        hostAPI = host_apis[device['hostApi']]

        if device['maxInputChannels'] > 0:
            audio_input_devices.append({"kind": "audioinput", "index": deviceIndex, "name": deviceName, "hostAPI": hostAPI})
        if device['maxOutputChannels'] > 0:
            audio_output_devices.append({"kind": "audiooutput", "index": deviceIndex, "name": deviceName, "hostAPI": hostAPI})
    audio_devices["audio_input_devices"] = audio_input_devices
    audio_devices["audio_output_devices"] = audio_output_devices

    json_compatible_item_data = jsonable_encoder(audio_devices)

    print(json_compatible_item_data)
