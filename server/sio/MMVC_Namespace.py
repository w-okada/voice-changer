import struct
from datetime import datetime
import numpy as np
import socketio
from voice_changer.VoiceChangerManager import VoiceChangerManager

import asyncio


class MMVC_Namespace(socketio.AsyncNamespace):
    sid: int = 0

    async def emitTo(self, data):
        timestamp = 0
        audio1 = np.zeros(1, dtype=np.float32)
        bin = struct.pack("<%sf" % len(audio1), *audio1)
        perf = data

        await self.emit("response", [timestamp, bin, perf], to=self.sid)

    def emit_coroutine(self, data):
        asyncio.run(self.emitTo(data))

    def __init__(self, namespace: str, voiceChangerManager: VoiceChangerManager):
        super().__init__(namespace)
        self.voiceChangerManager = voiceChangerManager
        # self.voiceChangerManager.voiceChanger.emitTo = self.emit_coroutine
        self.voiceChangerManager.setEmitTo(self.emit_coroutine)

    @classmethod
    def get_instance(cls, voiceChangerManager: VoiceChangerManager):
        if not hasattr(cls, "_instance"):
            cls._instance = cls("/test", voiceChangerManager)
        return cls._instance

    def on_connect(self, sid, environ):
        self.sid = sid
        print("[{}] connet sid : {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sid))
        pass

    async def on_request_message(self, sid, msg):
        self.sid = sid
        timestamp, data = msg
        # Receive and send int16 instead of float32 to reduce bandwidth requirement over websocket
        input_audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768

        out_audio, perf = self.voiceChangerManager.changeVoice(input_audio)
        out_audio = (out_audio * 32767).astype(np.int16).tobytes()
        await self.emit("response", [timestamp, out_audio, perf], to=sid)

    def on_disconnect(self, sid):
        # print('[{}] disconnect'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        pass
