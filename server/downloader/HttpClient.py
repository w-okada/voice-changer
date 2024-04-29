import requests

class HttpClient:
    _instance = None

    def __init__(self):
        self.session = requests.Session()

    @classmethod
    def get_client(cls) -> requests.Session:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance.session
