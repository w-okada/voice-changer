import aiohttp

class HttpClient:
    _instance = None

    def __init__(self):
        self.session: aiohttp.ClientSession = None

    @classmethod
    async def get_client(cls) -> aiohttp.ClientSession:
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.session = aiohttp.ClientSession()
        return cls._instance.session
