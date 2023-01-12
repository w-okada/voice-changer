import logging

# logging.getLogger('numba').setLevel(logging.WARNING)

class UvicornSuppressFilter(logging.Filter):
    def filter(self, record):
        return False

logger = logging.getLogger("uvicorn.error")
logger.addFilter(UvicornSuppressFilter())
# logger.propagate = False
logger = logging.getLogger("multipart.multipart")
logger.propagate = False

logging.getLogger('asyncio').setLevel(logging.WARNING)
