import struct
from datetime import datetime
import numpy as np
import socketio
from voice_changer.VoiceChangerManager import VoiceChangerManager


class MMVC_Namespace(socketio.AsyncNamespace):
    sid: int = 0

    def __init__(self, namespace: str, voiceChangerManager: VoiceChangerManager):
        super().__init__(namespace)
        self.voiceChangerManager = voiceChangerManager

    @classmethod
    def get_instance(cls, voiceChangerManager: VoiceChangerManager):
        if not hasattr(cls, "_instance"):
            cls._instance = cls("/test", voiceChangerManager)
        return cls._instance

    def on_connect(self, sid, environ):
        self.sid = sid
        print(
            "[{}] connet sid : {}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sid
            )
        )
        pass

    async def on_request_message(self, sid, msg):
        self.sid = sid
        timestamp = int(msg[0])
        data = msg[1]
        if isinstance(data, str):
            print(type(data))
            print(data)
            await self.emit("response", [timestamp, 0], to=sid)
        else:
            unpackedData = np.array(
                struct.unpack("<%sh" % (len(data) // struct.calcsize("<h")), data)
            ).astype(np.int16)

            res = self.voiceChangerManager.changeVoice(unpackedData)
            audio1 = res[0]
            perf = res[1] if len(res) == 2 else [0, 0, 0]
            bin = struct.pack("<%sh" % len(audio1), *audio1)
            await self.emit("response", [timestamp, bin, perf], to=sid)

    def on_disconnect(self, sid):
        # print('[{}] disconnect'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        pass
