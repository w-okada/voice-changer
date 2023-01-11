from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
class MMVC_Rest_Hello:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/api/hello", self.hello, methods=["GET"])

    def hello(self):
        return {"result": "Index"}



