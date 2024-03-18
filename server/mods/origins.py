from typing import Optional, Sequence
from urllib.parse import urlparse

ENFORCE_URL_ORIGIN_FORMAT = "Input origins must be well-formed URLs, i.e. https://google.com or https://www.google.com."
SCHEMAS = ('http', 'https')
LOCAL_ORIGINS = ('127.0.0.1', 'localhost')

def compute_local_origins(port: Optional[int] = None) -> list[str]:
    local_origins = [f'{schema}://{origin}' for schema in SCHEMAS for origin in LOCAL_ORIGINS]
    if port is not None:
        local_origins = [f'{origin}:{port}' for origin in local_origins]
    return local_origins


def normalize_origins(origins: Sequence[str]) -> set[str]:
    allowed_origins = set()
    for origin in origins:
        url = urlparse(origin)
        assert url.scheme, ENFORCE_URL_ORIGIN_FORMAT
        valid_origin = f'{url.scheme}://{url.hostname}'
        if url.port:
            valid_origin += f':{url.port}'
        allowed_origins.add(valid_origin)
    return allowed_origins
