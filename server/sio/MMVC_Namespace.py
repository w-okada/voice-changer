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
        # print("on_request_message", torch.cuda.memory_allocated())
        gpu = int(msg[0])
        srcId = int(msg[1])
        dstId = int(msg[2])
        timestamp = int(msg[3])
        convertChunkNum = int(msg[4])
        crossFadeLowerValue = msg[5]
        crossFadeOffsetRate = msg[6]
        crossFadeEndRate = msg[7]
        data = msg[8]
        # print(srcId, dstId, timestamp, convertChunkNum, crossFadeLowerValue, crossFadeOffsetRate, crossFadeEndRate)
        unpackedData = np.array(struct.unpack(
            '<%sh' % (len(data) // struct.calcsize('<h')), data))
        # audio1 = self.voiceChangerManager.changeVoice(
            # gpu, srcId, dstId, timestamp, prefixChunkSize, unpackedData)
        audio1 = self.voiceChangerManager.changeVoice(
            gpu, srcId, dstId, timestamp, convertChunkNum, crossFadeLowerValue, crossFadeOffsetRate, crossFadeEndRate, unpackedData)
        
        print("sio result:", len(audio1), audio1.shape)
        bin = struct.pack('<%sh' % len(audio1), *audio1)
        await self.emit('response', [timestamp, bin])

    def on_disconnect(self, sid):
        # print('[{}] disconnect'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        pass

