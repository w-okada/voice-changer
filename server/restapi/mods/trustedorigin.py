from typing import Optional, Sequence, Literal

from mods.origins import compute_local_origins, normalize_origins
from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class TrustedOriginMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        allowed_origins: Optional[Sequence[str]] = None,
        port: Optional[int] = None,
    ) -> None:
        self.allowed_origins: set[str] = set()

        self.any_origin = '*' in allowed_origins
        if not self.any_origin:
            local_origins = compute_local_origins(port)
            self.allowed_origins.update(local_origins)

            if allowed_origins is not None:
                normalized_origins = normalize_origins(allowed_origins)
                self.allowed_origins.update(normalized_origins)

        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in (
            "http",
            "websocket",
        ):  # pragma: no cover
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        origin = headers.get("origin", "")
        # Origin header is not present for same origin
        if not origin or self.any_origin or origin in self.allowed_origins:
            await self.app(scope, receive, send)
            return

        response = PlainTextResponse("Invalid origin header", status_code=400)
        await response(scope, receive, send)
