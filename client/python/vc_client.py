import argparse
import pyaudio
import wave
import struct
import socketio
import ssl
from datetime import datetime
import time

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import signal
import sys
import numpy as np

BUFFER_SIZE = 2048 * 2 * 10
SAMPLING_RATE = 44100
GAIN = 10


def setupArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:18888", help="url")
    parser.add_argument("--input", type=int, required=True, help="input device index")
    parser.add_argument("--output", type=int, default=-1, help="input device index")
    parser.add_argument("--to", type=str, default="", help="sid")

    return parser


class MockStream:

    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate
        self.start_count = 2
        self.end_count = 2
        self.fr = None
        self.fw = None

    def open_inputfile(self, input_filename):
        self.fr = wave.open(input_filename, 'rb')

    def open_outputfile(self, output_filename):
        self.fw = wave.open(output_filename, 'wb')
        self.fw.setnchannels(1)
        self.fw.setsampwidth(2)
        self.fw.setframerate(self.sampling_rate)

    def read(self, length, exception_on_overflow=False):
        if self.start_count > 0:
            wav = bytes(length * 2)
            self.start_count -= 1
        else:
            wav = self.fr.readframes(length)
        if len(wav) <= 0:
            wav = bytes(length * 2)
            self.end_count -= 1
            if self.end_count < 0:
                Hyperparameters.VC_END_FLAG = True
        return wav

    def write(self, wav):
        self.fw.writeframes(wav)

    def stop_stream(self):
        pass

    def close(self):
        if self.fr != None:
            self.fr.close()
            self.fr = None
        if self.fw != None:
            self.fw.close()
            self.fw = None


class MyCustomNamespace(socketio.ClientNamespace):
    def __init__(self, namespace: str, audio_output_stream, file_output_stream):
        super().__init__(namespace)
        self.audio_output_stream = audio_output_stream
        self.file_output_stream = file_output_stream

    def on_connect(self):
        print(f'connected')

    def on_disconnect(self):
        print(f'disconnected')

    def on_response(self, msg):
        timestamp = msg[0]
        responseTime = time.time() * 1000 - timestamp
        data = msg[1]
        perf = msg[2]
        print(f"RT:{responseTime}msec", perf)
        unpackedData = struct.unpack('<%sh' % (len(data) // struct.calcsize('<h')), data)
        data = np.array(unpackedData)
        data = data * GAIN
        data = struct.pack('<%sh' % len(data), *data)

        if self.file_output_stream != None:
            self.file_output_stream.write(data)
        if self.audio_output_stream != None:
            self.audio_output_stream.write(data)


if __name__ == '__main__':
    parser = setupArgParser()
    args, unknown = parser.parse_known_args()

    url = args.url
    inputDevice = args.input
    outputDevice = args.output
    to = args.to

    audio = pyaudio.PyAudio()
    audio_input_stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLING_RATE,
        frames_per_buffer=BUFFER_SIZE,
        input_device_index=inputDevice,
        input=True)

    print("output device", outputDevice)
    if outputDevice >= 0:
        audio_output_stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLING_RATE,
            frames_per_buffer=BUFFER_SIZE,
            output_device_index=outputDevice,
            output=True)
    else:
        audio_output_stream = None

    # mock_stream_out = MockStream(24000)
    # mock_stream_out.open_outputfile("test.wav")
    mock_stream_out = None

    # mock_stream_in = MockStream(24000)
    # mock_stream_in.open_outputfile("test_in.wav")

    my_namespace = MyCustomNamespace("/test", audio_output_stream, mock_stream_out)

    sio = socketio.Client(ssl_verify=False)
    sio.register_namespace(my_namespace)
    sio.connect(url)
    try:
        while True:
            in_wav = audio_input_stream.read(BUFFER_SIZE, exception_on_overflow=False)
            sio.emit('request_message', [time.time() * 1000, in_wav], namespace="/test")
    except KeyboardInterrupt:
        audio_input_stream.stop_stream()
        audio_input_stream.close()
        audio_output_stream.stop_stream()
        audio_output_stream.close()
        audio.terminate()

        mock_stream_out.close()
