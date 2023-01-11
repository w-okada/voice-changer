import struct
from datetime import datetime
import numpy as np
import socketio
from voice_changer.VoiceChangerManager import VoiceChangerManager

class MMVC_Namespace(socketio.AsyncNamespace):
    def __init__(self, namespace:str, voiceChangerManager:VoiceChangerManager):
        super().__init__(namespace)
        self.voiceChangerManager = voiceChangerManager

    @classmethod
    def get_instance(cls, voiceChangerManager:VoiceChangerManager):
        if not hasattr(cls, "_instance"):
            cls._instance = cls("/test", voiceChangerManager)
        return cls._instance

    def on_connect(self, sid, environ):
        # print('[{}] connet sid : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S') , sid))
        pass

    async def on_request_message(self, sid, msg):
        timestamp = int(msg[0])
        data = msg[1]

        if(isinstance(data, str)):
            print(type(data))
            print(data)
            await self.emit('response', [timestamp, 0])
        else:
            unpackedData = np.array(struct.unpack('<%sh' % (len(data) // struct.calcsize('<h')), data))
            audio1 = self.voiceChangerManager.changeVoice(unpackedData)
            # print("sio result:", len(audio1), audio1.shape)
            bin = struct.pack('<%sh' % len(audio1), *audio1)
            await self.emit('response', [timestamp, bin])

    def on_disconnect(self, sid):
        # print('[{}] disconnect'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        pass

