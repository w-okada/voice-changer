from restapi.mods.trustedorigin import TrustedOriginMiddleware
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from typing import Callable, Optional, Sequence
from mods.log_control import VoiceChangaerLogger
from voice_changer.VoiceChangerManager import VoiceChangerManager

from restapi.MMVC_Rest_Hello import MMVC_Rest_Hello
from restapi.MMVC_Rest_VoiceChanger import MMVC_Rest_VoiceChanger
from restapi.MMVC_Rest_Fileuploader import MMVC_Rest_Fileuploader
from const import UPLOAD_DIR, FRONTEND_DIR, TMP_DIR

logger = VoiceChangaerLogger.get_instance().getLogger()


class ValidationErrorLoggingRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            try:
                return await original_route_handler(request)
            except RequestValidationError as exc:  # type: ignore
                print("Exception", request.url, str(exc))
                body = await request.body()
                detail = {"errors": exc.errors(), "body": body.decode()}
                raise HTTPException(status_code=422, detail=detail)

        return custom_route_handler


class MMVC_Rest:
    _instance = None

    @classmethod
    def get_instance(
        cls,
        voiceChangerManager: VoiceChangerManager,
        model_dir: str,
        allowedOrigins: Optional[Sequence[str]] = None,
        port: Optional[int] = None,
    ):
        if cls._instance is None:
            logger.info("[Voice Changer] MMVC_Rest initializing...")
            app_fastapi = FastAPI()
            app_fastapi.router.route_class = ValidationErrorLoggingRoute
            app_fastapi.add_middleware(
                TrustedOriginMiddleware,
                allowed_origins=allowedOrigins,
                port=port
            )

            app_fastapi.mount("/tmp", StaticFiles(directory=TMP_DIR), name="static")
            app_fastapi.mount("/upload_dir", StaticFiles(directory=UPLOAD_DIR), name="static")

            app_fastapi.mount(
                "/model_dir",
                StaticFiles(directory=model_dir),
                name="static",
            )

            restHello = MMVC_Rest_Hello()
            app_fastapi.include_router(restHello.router)
            restVoiceChanger = MMVC_Rest_VoiceChanger(voiceChangerManager)
            app_fastapi.include_router(restVoiceChanger.router)
            fileUploader = MMVC_Rest_Fileuploader(voiceChangerManager)
            app_fastapi.include_router(fileUploader.router)

            cls._instance = app_fastapi
            logger.info("[Voice Changer] MMVC_Rest initializing... done.")
            return cls._instance

        return cls._instance
