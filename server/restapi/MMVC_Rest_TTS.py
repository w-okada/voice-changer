"""TTS 連携 (Read Aloud などの OpenAI互換クライアント向け)。

  POST /v1/audio/speech (/audio/speech)
    -> Edge-TTS で文章を音声化
    -> ロード済みRVCモデルで文全体を1回変換 (VoiceChangerManager.convert_offline)
    -> mp3 / wav を返す
  GET  /tts           設定・テスト用ページ
  GET  /v1/models     接続テスト用 (Read Aloud の保存時テストが叩く)

安全方針: すべてのTTS系ルートは loopback からのアクセスのみ許可する。
拡張機能など別オリジンからの利用は --tts-allowed-origins で明示許可したものだけ (trustedorigin.py)。
"""
import asyncio
import hashlib
import io
import ipaddress
import json
import os
import re
import shutil
import tempfile
import threading
import time
import wave
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from mods.log_control import VoiceChangaerLogger

logger = VoiceChangaerLogger.get_instance().getLogger()

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "tts_settings.json"
UI_PATH = Path(__file__).resolve().parent / "tts_ui" / "index.html"

MAX_CHARS = 1000
CHUNK_CHARS = 120  # 1回のTTS+RVC変換にかける文字数の目安 (長文は文単位に分割して順に処理)
MAX_QUEUE = 8
MAX_AUDIO_SEC = 40  # 1チャンクあたりの上限
MAX_TOTAL_SEC = 300  # 1リクエスト全体の上限
GAP_SEC = 0.15  # チャンク間の無音
FALLBACK_ON_SILENT_RVC = True  # RVC出力が無音のチャンクは、変換前の音声で代用して読み上げを止めない
MAX_SILENT_STREAK = 3  # RVC出力が無音の変換がこの回数連続したら、設定異常とみなしてエラーにする
MAX_BODY_BYTES = 64 * 1024
MAX_MP3_BYTES = 8 * 1024 * 1024
CACHE_MAX = 32
TTS_TIMEOUT_SEC = 30
SILENCE_THRESHOLD = 200  # int16。これ以下は無音扱いで前後をトリム
INTERNAL_RATE = 24000  # Edge-TTS (mp3 24kHz) のデコード先
VOICE_RE = re.compile(r"^[A-Za-z0-9\-]{5,64}$")
LOCALE_RE = re.compile(r"^[A-Za-z\-]{0,12}$")
DEFAULT_SETTINGS = {"edge_voice": "ja-JP-NanamiNeural", "speed": 1.0, "rvc": True}
FALLBACK_VOICES = [
    {"name": "ja-JP-NanamiNeural", "locale": "ja-JP", "gender": "Female"},
    {"name": "ja-JP-KeitaNeural", "locale": "ja-JP", "gender": "Male"},
]
UI_CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
    "connect-src 'self'; media-src blob:; img-src 'self' data:; base-uri 'none'; form-action 'none'"
)


_SENT_END = re.compile(r"(?<=[。！？!?\n])")
_SOFT_BREAK = re.compile(r"(?<=[、，,;；:：\s])")


def split_text(text: str, limit: int = CHUNK_CHARS) -> list:
    """文単位に分け、limit 文字以内になるよう詰め直す。長い文は読点などで、無ければ強制的に分割する。"""
    sentences = []
    for sent in _SENT_END.split(text):
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= limit:
            sentences.append(sent)
            continue
        buf = ""
        for part in _SOFT_BREAK.split(sent):
            if len(buf) + len(part) <= limit:
                buf += part
                continue
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
            while len(part) > limit:
                sentences.append(part[:limit])
                part = part[limit:]
            buf = part
        if buf.strip():
            sentences.append(buf.strip())
    chunks, cur = [], ""
    for sent in sentences:
        if cur and len(cur) + len(sent) > limit:
            chunks.append(cur)
            cur = ""
        cur += sent
    if cur:
        chunks.append(cur)
    return chunks


class TTSError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class Unspeakable(Exception):
    """読み上げ可能な内容が無い (記号だけ等)。無音を返す"""


def _err(status: int, message: str) -> JSONResponse:
    # Read Aloud は `res.json().error` を投げるので OpenAI 形式で返す
    return JSONResponse({"error": {"message": message, "type": "tts_error"}}, status_code=status)


def _is_local(request: Request) -> bool:
    client = request.client
    if client is None:
        return False
    try:
        ip = ipaddress.ip_address(client.host)
    except ValueError:
        return client.host == "localhost"
    if getattr(ip, "ipv4_mapped", None) is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback


def _validate_settings(src: dict) -> dict:
    """部分更新可。不正値は 400。bool は数値として扱わない。"""
    out = {}
    if "edge_voice" in src:
        v = src["edge_voice"]
        if not isinstance(v, str) or not VOICE_RE.match(v):
            raise TTSError(400, "edge_voice が不正です")
        out["edge_voice"] = v
    if "speed" in src:
        v = src["speed"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.5 <= float(v) <= 2.0:
            raise TTSError(400, "speed は 0.5〜2.0 の数値で指定してください")
        out["speed"] = float(v)
    if "rvc" in src:
        if not isinstance(src["rvc"], bool):
            raise TTSError(400, "rvc は true/false で指定してください")
        out["rvc"] = src["rvc"]
    return out


class MMVC_Rest_TTS:
    def __init__(self, voiceChangerManager):
        self.manager = voiceChangerManager
        self.router = APIRouter()
        self.router.add_api_route("/tts", self.ui, methods=["GET"])
        self.router.add_api_route("/tts/api/settings", self.get_settings, methods=["GET"])
        self.router.add_api_route("/tts/api/settings", self.post_settings, methods=["POST"])
        self.router.add_api_route("/tts/api/status", self.status, methods=["GET"])
        self.router.add_api_route("/tts/api/voices", self.voices, methods=["GET"])
        for path in ("/models", "/v1/models"):
            self.router.add_api_route(path, self.models, methods=["GET"])
        for path in ("/audio/speech", "/v1/audio/speech"):
            self.router.add_api_route(path, self.speech, methods=["POST"])

        self._settings = self._load_settings()
        self._settings_lock = threading.Lock()
        self._cache: "OrderedDict[str, bytes]" = OrderedDict()
        self._waiting = 0
        self._rvc_lock = asyncio.Lock()
        self._silent_streak = 0
        self._voices_cache: tuple = (0.0, [])

    # ------------------------------------------------------------ 設定の保存
    def _load_settings(self) -> dict:
        settings = dict(DEFAULT_SETTINGS)
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                settings.update(_validate_settings(data))
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TTSError):
            logger.warning("[TTS] tts_settings.json を読めないので既定値を使います")
        return settings

    def _save_settings(self) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(SETTINGS_PATH.parent), prefix=".tts_settings.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._settings, f, ensure_ascii=False, indent=2)
            os.replace(tmp, SETTINGS_PATH)  # 書き込み途中で壊れないようアトミックに置換
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------ 共通
    def _model_ready(self) -> bool:
        vc = getattr(self.manager, "voiceChanger", None)
        model = getattr(vc, "voiceChanger", None)
        return model is not None and hasattr(model, "inference_offline")

    def _fingerprint(self) -> str:
        """モデル/変換設定が変わったらキャッシュを無効にするための識別子"""
        try:
            model = self.manager.voiceChanger.voiceChanger
            slot, st = getattr(model, "slotInfo", None), getattr(model, "settings", None)
            parts = [id(model), getattr(slot, "slotIndex", None), getattr(slot, "modelFile", None)]
            parts += [getattr(st, k, None) for k in ("tran", "indexRatio", "protect", "dstId")]
            return repr(parts)
        except AttributeError:
            return "none"

    @staticmethod
    async def _read_json(request: Request) -> dict:
        length = request.headers.get("content-length")
        try:
            if length is not None and int(length) > MAX_BODY_BYTES:
                raise TTSError(413, "request body too large")
        except ValueError:
            raise TTSError(400, "invalid content-length")
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            raise TTSError(413, "request body too large")
        try:
            data = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise TTSError(400, "invalid JSON")
        if not isinstance(data, dict):
            raise TTSError(400, "JSON object expected")
        return data

    # ------------------------------------------------------------ ルート: ページ/設定
    async def ui(self, request: Request):
        if not _is_local(request):
            return _err(403, "local access only")
        if not UI_PATH.is_file():
            return _err(404, "tts_ui/index.html not found")
        return FileResponse(
            str(UI_PATH), media_type="text/html",
            headers={"Content-Security-Policy": UI_CSP, "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    async def get_settings(self, request: Request):
        if not _is_local(request):
            return _err(403, "local access only")
        return JSONResponse(self._settings)

    async def post_settings(self, request: Request):
        if not _is_local(request):
            return _err(403, "local access only")
        try:
            update = _validate_settings(await self._read_json(request))
            with self._settings_lock:
                self._settings.update(update)
                self._save_settings()
        except TTSError as e:
            return _err(e.status, e.message)
        except OSError:
            return _err(500, "設定ファイルを保存できませんでした")
        return JSONResponse(self._settings)

    async def status(self, request: Request):
        if not _is_local(request):
            return _err(403, "local access only")
        try:
            import edge_tts  # noqa: F401
            edge_ok = True
        except ImportError:
            edge_ok = False
        return JSONResponse({"model_ready": self._model_ready(), "edge_tts": edge_ok, "ffmpeg": shutil.which("ffmpeg") is not None})

    async def voices(self, request: Request):
        if not _is_local(request):
            return _err(403, "local access only")
        locale = request.query_params.get("locale", "")
        if not LOCALE_RE.match(locale):
            return _err(400, "invalid locale")
        now = time.monotonic()
        stamp, cached = self._voices_cache
        fallback = False
        if cached and now - stamp < 3600:
            voices = cached
        else:
            try:
                import edge_tts
                raw = await asyncio.wait_for(edge_tts.list_voices(), timeout=15)
                voices = sorted(
                    ({"name": v["ShortName"], "locale": v["Locale"], "gender": v.get("Gender", "")} for v in raw),
                    key=lambda v: (v["locale"], v["name"]),
                )
                self._voices_cache = (now, voices)
            except Exception:  # オフライン/未インストール時は固定リストで継続
                voices, fallback = FALLBACK_VOICES, True
        if locale:
            voices = [v for v in voices if v["locale"].lower().startswith(locale.lower())]
        return JSONResponse({"voices": voices, "fallback": fallback})

    async def models(self, request: Request):
        if not _is_local(request):
            return _err(403, "local access only")
        # Read Aloud の保存時テストが GET {URL}/models を叩く
        return JSONResponse({"object": "list", "data": [{"id": "rvc-tts", "object": "model", "owned_by": "local"}]})

    # ------------------------------------------------------------ ルート: 音声合成
    async def speech(self, request: Request):
        if not _is_local(request):
            return _err(403, "local access only")
        try:
            body = await self._read_json(request)
            text = body.get("input")
            if not isinstance(text, str) or not text.strip():
                raise TTSError(400, "input is empty")
            text = text.strip()
            if len(text) > MAX_CHARS:
                raise TTSError(400, f"input too long (max {MAX_CHARS} chars)")
            fmt = str(body.get("response_format") or "mp3").lower()
            if fmt not in ("mp3", "wav"):
                raise TTSError(400, "response_format must be mp3 or wav")
            opts = _validate_settings({k: body[k] for k in ("edge_voice", "speed", "rvc") if body.get(k) is not None})
            with self._settings_lock:
                merged = {**self._settings, **opts}
            data = await self._synthesize(text, merged["edge_voice"], merged["speed"], merged["rvc"], fmt)
        except TTSError as e:
            return _err(e.status, e.message)
        return Response(data, media_type="audio/mpeg" if fmt == "mp3" else "audio/wav")

    async def _synthesize(self, text: str, voice: str, speed: float, rvc: bool, fmt: str) -> bytes:
        if rvc and not self._model_ready():
            raise TTSError(503, "RVCモデルがロードされていません (voice-changer のGUIでモデルを選択してください)")
        key = hashlib.sha256(
            json.dumps([text, voice, speed, rvc, fmt, self._fingerprint() if rvc else ""]).encode()
        ).hexdigest()
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        if self._waiting >= MAX_QUEUE:
            raise TTSError(429, "too many pending requests")

        self._waiting += 1
        started = time.monotonic()
        try:
            parts, total, fallbacks = [], 0.0, 0
            chunks = split_text(text)
            for chunk in chunks:
                try:
                    pcm, rate = await self._synth_edge(chunk, voice, speed)
                    if pcm.size == 0 or int(np.abs(pcm.astype(np.int32)).max()) <= SILENCE_THRESHOLD:
                        raise Unspeakable()
                    if pcm.size > MAX_AUDIO_SEC * rate:
                        raise TTSError(413, f"generated audio too long (max {MAX_AUDIO_SEC}s per chunk)")
                    if rvc:
                        src_pcm, src_rate = pcm, rate
                        pcm, rate = await self._convert(src_pcm, src_rate)
                        peak = int(np.abs(pcm.astype(np.int32)).max()) if pcm.size else 0
                        if peak <= SILENCE_THRESHOLD:
                            self._silent_streak += 1
                            logger.warning(
                                "[TTS] RVC出力が無音 (chars=%d in=%.2fs in_peak=%d out=%.2fs out_peak=%d streak=%d)",
                                len(chunk), src_pcm.size / src_rate, int(np.abs(src_pcm.astype(np.int32)).max()),
                                pcm.size / rate, peak, self._silent_streak,
                            )
                            if not FALLBACK_ON_SILENT_RVC or self._silent_streak >= MAX_SILENT_STREAK:
                                raise TTSError(502, "RVCの出力が無音です (モデル/ピッチ設定を確認してください)")
                            pcm, rate = src_pcm, src_rate  # 変換前の音声で代用
                            fallbacks += 1
                        else:
                            self._silent_streak = 0
                    pcm = self._trim_and_fade(pcm, rate)
                except Unspeakable:
                    continue  # 記号だけの断片などは飛ばす
                total += pcm.size / rate
                if total > MAX_TOTAL_SEC:
                    raise TTSError(413, f"generated audio too long (max {MAX_TOTAL_SEC}s)")
                parts.append((pcm, rate))
            if parts:
                rate = max(r for _, r in parts)  # 代用音声と変換後でSRが違う場合は高い方に揃える
                gap = np.zeros(int(GAP_SEC * rate), dtype=np.int16)
                joined = []
                for i, (p, r) in enumerate(parts):
                    if i:
                        joined.append(gap)
                    joined.append(self._resample(p, r, rate))
                pcm = np.concatenate(joined)
            else:
                logger.warning("[TTS] 読み上げ不可の入力 (chars=%d) -> 無音を返します", len(text))
                rate = INTERNAL_RATE
                pcm = np.zeros(int(0.2 * rate), dtype=np.int16)
            data = await self._encode(pcm, rate, fmt)
        finally:
            self._waiting -= 1

        logger.info("[TTS] ok chars=%d chunks=%d rvc=%s fallbacks=%d audio=%.2fs took=%.2fs", len(text), len(chunks), rvc, fallbacks, pcm.size / rate, time.monotonic() - started)
        self._cache[key] = data
        while len(self._cache) > CACHE_MAX:
            self._cache.popitem(last=False)
        return data

    # ------------------------------------------------------------ パイプライン部品
    @staticmethod
    async def _ffmpeg(args: list, data: bytes, what: str) -> bytes:
        if shutil.which("ffmpeg") is None:
            raise TTSError(502, "ffmpeg が見つかりません (pacman -S ffmpeg)")
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error", *args,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(data), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise TTSError(502, f"ffmpeg ({what}) がタイムアウトしました")
        if proc.returncode != 0 or not out:
            raise TTSError(502, f"ffmpeg ({what}) 失敗: {err.decode(errors='replace')[:120]}")
        return out

    async def _synth_edge(self, text: str, voice: str, speed: float):
        try:
            import aiohttp
            import edge_tts
            from edge_tts.exceptions import EdgeTTSException, NoAudioReceived
        except ImportError:
            raise TTSError(503, "edge-tts が未インストールです (pip install edge-tts)")
        rate = f"{round((speed - 1.0) * 100):+d}%"

        async def collect() -> bytes:
            buf = bytearray()
            async for chunk in edge_tts.Communicate(text, voice, rate=rate).stream():
                if chunk.get("type") == "audio":
                    buf.extend(chunk["data"])
                    if len(buf) > MAX_MP3_BYTES:  # 暴走防止
                        raise TTSError(413, "generated audio too large")
            return bytes(buf)

        try:
            mp3 = await asyncio.wait_for(collect(), timeout=TTS_TIMEOUT_SEC)
        except NoAudioReceived:
            raise Unspeakable()
        except (asyncio.TimeoutError, OSError, aiohttp.ClientError, EdgeTTSException, ValueError) as e:
            raise TTSError(502, f"Edge-TTS に接続/応答できません ({type(e).__name__})")
        if not mp3:
            raise Unspeakable()
        out = await self._ffmpeg(["-i", "pipe:0", "-f", "s16le", "-ar", str(INTERNAL_RATE), "-ac", "1", "pipe:1"], mp3, "decode")
        return np.frombuffer(out, dtype="<i2").astype(np.int16), INTERNAL_RATE

    async def _convert(self, pcm: np.ndarray, rate: int):
        async with self._rvc_lock:  # GPU同時実行を避けて直列化
            try:
                return await asyncio.to_thread(self.manager.convert_offline, pcm, rate)
            except ValueError as e:
                raise TTSError(400, str(e))
            except RuntimeError as e:  # モデル未ロード / GPUメモリ不足 など
                raise TTSError(503, f"RVC変換に失敗しました: {str(e)[:160]}")
            except Exception as e:
                logger.error("[TTS] convert failed: %s", e)
                raise TTSError(500, f"RVC変換に失敗しました: {str(e)[:160]}")

    @staticmethod
    def _resample(pcm: np.ndarray, src: int, dst: int) -> np.ndarray:
        """線形補間の簡易リサンプル (代用音声のSRを揃えるためだけに使う)"""
        if src == dst or pcm.size == 0:
            return pcm
        n = int(round(pcm.size * dst / src))
        x = np.interp(np.linspace(0, pcm.size - 1, n), np.arange(pcm.size), pcm.astype(np.float32))
        return np.clip(x, -32768, 32767).astype(np.int16)

    @staticmethod
    def _trim_and_fade(pcm: np.ndarray, rate: int) -> np.ndarray:
        loud = np.flatnonzero(np.abs(pcm.astype(np.int32)) > SILENCE_THRESHOLD)
        if loud.size == 0:
            raise TTSError(502, "RVCの出力が無音です (モデル/ピッチ設定を確認してください)")
        margin, fade = int(0.05 * rate), int(0.01 * rate)
        start = max(int(loud[0]) - margin, 0)
        end = min(int(loud[-1]) + margin + 1, pcm.size)
        seg = pcm[start:end].astype(np.float32)
        f = min(fade, seg.size // 2)
        if f > 0:
            seg[:f] *= np.linspace(0.0, 1.0, f, endpoint=False, dtype=np.float32)
            seg[-f:] *= np.linspace(1.0, 0.0, f, endpoint=False, dtype=np.float32)
        return np.clip(seg, -32768, 32767).astype(np.int16)

    async def _encode(self, pcm: np.ndarray, rate: int, fmt: str) -> bytes:
        if fmt == "wav":
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(pcm.astype("<i2").tobytes())
            return buf.getvalue()
        return await self._ffmpeg(
            ["-f", "s16le", "-ar", str(rate), "-ac", "1", "-i", "pipe:0", "-f", "mp3", "-b:a", "128k", "pipe:1"],
            pcm.astype("<i2").tobytes(), "encode",
        )
