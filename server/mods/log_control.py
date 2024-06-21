import logging
import traceback
import sys
from const import LOG_FILE

class UvicornSuppressFilter(logging.Filter):
    def filter(self, record):
        return False


class NullHandler(logging.Handler):
    def emit(self, record):
        pass


class DebugStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            super().emit(record)
        except Exception as e:
            print(f"Error logging message: {e}", file=sys.stderr)
            traceback.print_exc()


class DebugFileHandler(logging.FileHandler):
    def emit(self, record):
        try:
            super().emit(record)
        except Exception as e:
            print(f"Error writing log message to file: {e}", file=sys.stderr)
            traceback.print_exc()


class VoiceChangaerLogger:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # logger = logging.getLogger("uvicorn.error")
        # logger.addFilter(UvicornSuppressFilter())

        # logging.basicConfig(filename='myapp.log', level=logging.INFO)
        # logging.basicConfig(level=logging.NOTSET)
        logging.root.handlers = [NullHandler()]

        logger = logging.getLogger("fairseq.tasks.hubert_pretraining")
        logger.addFilter(UvicornSuppressFilter())

        logger = logging.getLogger("fairseq.models.hubert.hubert")
        logger.addFilter(UvicornSuppressFilter())

        logger = logging.getLogger("fairseq.tasks.text_to_speech")
        logger.addFilter(UvicornSuppressFilter())

        logger = logging.getLogger("numba.core.ssa")
        logger.addFilter(UvicornSuppressFilter())

        logger = logging.getLogger("numba.core.interpreter")
        logger.addFilter(UvicornSuppressFilter())

        logger = logging.getLogger("numba.core.byteflow")
        logger.addFilter(UvicornSuppressFilter())

        # logger.propagate = False

        logger = logging.getLogger("multipart.multipart")
        logger.propagate = False

        logging.getLogger("asyncio").setLevel(logging.WARNING)

        logger = logging.getLogger("vcclient")
        logger.setLevel(logging.DEBUG)
        self.logger = logger

    def initialize(self, initialize: bool):
        if not self.logger.handlers:
            if initialize:
                # file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w")
                file_handler = DebugFileHandler(LOG_FILE, encoding="utf-8", mode="w")
            else:
                # file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
                file_handler = DebugFileHandler(LOG_FILE, encoding="utf-8")
            file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s")
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)

            stream_formatter = logging.Formatter("%(message)s")
            # stream_handler = logging.StreamHandler()
            stream_handler = DebugStreamHandler()
            stream_handler.setFormatter(stream_formatter)
            stream_handler.setLevel(logging.INFO)
            self.logger.addHandler(stream_handler)

    def getLogger(self):
        return self.logger
