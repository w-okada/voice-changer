import struct
from datetime import datetime
import numpy as np
import socketio
from voice_changer.VoiceChangerManager import VoiceChangerManager

import time
import multiprocessing as mp


# Queueからデータを読み取り
def read(q_in, q_out):
    while True:
        time.sleep(1)
        [timestamp, sid] = q_in.get(True)
        print("put........................................")
        q_out.put([timestamp, sid])


class MMVC_Namespace(socketio.AsyncNamespace):
    def __init__(self, namespace: str, voiceChangerManager: VoiceChangerManager):
        super().__init__(namespace)
        self.voiceChangerManager = voiceChangerManager

        self.q_in = mp.Queue()
        self.q_out = mp.Queue()
        self.p = mp.Process(target=read, args=(self.q_in, self.q_out))
        self.p.start()

    @classmethod
    def get_instance(cls, voiceChangerManager: VoiceChangerManager):
        if not hasattr(cls, "_instance"):
            cls._instance = cls("/test", voiceChangerManager)
        return cls._instance

    def on_connect(self, sid, environ):
        print('[{}] connet sid : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), sid))
        pass

    async def on_request_message(self, sid, msg):
        timestamp = int(msg[0])
        data = msg[1]
        if (isinstance(data, str)):
            print(type(data))
            print(data)
            await self.emit('response', [timestamp, 0], to=sid)
        else:
            # print("receive ")
            # self.q_in.put([timestamp, sid])
            # while self.q_out.empty() == False:
            #     print("send........................................")
            #     [timestamp, sid] = self.q_out.get(True)
            #     await self.emit('response', [timestamp, 0], to=sid)
            # print("end")

            # print("receive ")
            # time.sleep(2)
            # await self.emit('response', [timestamp, 0], to=sid)
            # print("end")

            unpackedData = np.array(struct.unpack('<%sh' % (len(data) // struct.calcsize('<h')), data))
            audio1, perf = self.voiceChangerManager.changeVoice(unpackedData)
            bin = struct.pack('<%sh' % len(audio1), *audio1)
            await self.emit('response', [timestamp, bin, perf], to=sid)

    def on_disconnect(self, sid):
        # print('[{}] disconnect'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        pass
