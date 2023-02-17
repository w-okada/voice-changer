import argparse
import pyaudio
import wave
import struct
import socketio
import ssl
from datetime import datetime
import time

context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.verify_mode = ssl.CERT_NONE


def setupArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=int, default=18888, help="port")
    parser.add_argument("-d", type=int, help="device index")
    parser.add_argument("-s", type=str, default="", help="sid")

    return parser


class MockStream:
    """
    オーディオストリーミング入出力をファイル入出力にそのまま置き換えるためのモック
    """

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
            self.start_count -= 1  # 最初の2回はダミーの空データ送る
        else:
            wav = self.fr.readframes(length)
        if len(wav) <= 0:  # データなくなってから最後の2回はダミーの空データを送る
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


mock_stream_out = MockStream(24000)
mock_stream_out.open_outputfile("test.wav")


class MyCustomNamespace(socketio.ClientNamespace):  # 名前空間を設定するクラス
    def on_connect(self):
        print('[{}] connect'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    def on_disconnect(self):
        print('[{}] disconnect'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    def on_response(self, msg):
        print('[{}] response : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))
        timestamp = msg[0]
        data = msg[1]
        unpackedData = struct.unpack('<%sh' % (len(data) // struct.calcsize('<h')), data)
        mock_stream_out.write(data)


def my_background_task(sio):  # ここにバックグランド処理のコードを書く
    while True:

        sio.emit('broadcast_message', "aaa", namespace="/test")  # ターミナルで入力された文字をサーバーに送信
        sio.sleep(1)


if __name__ == '__main__':
    parser = setupArgParser()
    args, unknown = parser.parse_known_args()

    port = args.p
    deviceIndex = args.d
    sid = args.s

    audio = pyaudio.PyAudio()
    audio_input_stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24000,
        frames_per_buffer=4096,
        input_device_index=args.d,
        input=True)

    sio = socketio.Client(ssl_verify=False)
    sio.register_namespace(MyCustomNamespace("/test"))
    sio.connect("https://192.168.0.3:18888")
    while True:
        in_wav = audio_input_stream.read(4096, exception_on_overflow=False)
        bin = struct.pack('<%sh' % len(in_wav), *in_wav)
        sio.emit('request_message', [1000, bin], namespace="/test")
        # sio.start_background_task(my_background_task, sio)
