import logging

# logging.getLogger('numba').setLevel(logging.WARNING)


class UvicornSuppressFilter(logging.Filter):
    def filter(self, record):
        return False


class NullHandler(logging.Handler):
    def emit(self, record):
        pass


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

        if not logger.handlers:
            # pass
            # file_handler = logging.FileHandler('vvclient.log', encoding='utf-8', mode='w')
            file_handler = logging.FileHandler('vvclient.log', encoding='utf-8')
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s')
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)

            stream_formatter = logging.Formatter('%(message)s')
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(stream_formatter)
            stream_handler.setLevel(logging.INFO)
            logger.addHandler(stream_handler)

        self.logger = logger

    def getLogger(self):
        return self.logger


def setup_loggers(startMessage: str):
    pass

    # # logger = logging.getLogger("uvicorn.error")
    # # logger.addFilter(UvicornSuppressFilter())

    # logger = logging.getLogger("fairseq.tasks.hubert_pretraining")
    # logger.addFilter(UvicornSuppressFilter())

    # logger = logging.getLogger("fairseq.models.hubert.hubert")
    # logger.addFilter(UvicornSuppressFilter())

    # logger = logging.getLogger("fairseq.tasks.text_to_speech")
    # logger.addFilter(UvicornSuppressFilter())

    # logger = logging.getLogger("numba.core.ssa")
    # logger.addFilter(UvicornSuppressFilter())

    # logger = logging.getLogger("numba.core.interpreter")
    # logger.addFilter(UvicornSuppressFilter())

    # logger = logging.getLogger("numba.core.byteflow")
    # logger.addFilter(UvicornSuppressFilter())

    # # logger.propagate = False

    # logger = logging.getLogger("multipart.multipart")
    # logger.propagate = False

    # logging.getLogger("asyncio").setLevel(logging.WARNING)

    # logger = logging.getLogger("vcclient")
    # logger.setLevel(logging.DEBUG)

    # if not logger.handlers:
    #     # file_handler = logging.FileHandler('vvclient.log', encoding='utf-8', mode='w')
    #     file_handler = logging.FileHandler('vvclient.log', encoding='utf-8')
    #     file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s')
    #     file_handler.setFormatter(file_formatter)
    #     file_handler.setLevel(logging.INFO)
    #     logger.addHandler(file_handler)

    #     stream_formatter = logging.Formatter('%(message)s')
    #     stream_handler = logging.StreamHandler()
    #     stream_handler.setFormatter(stream_formatter)
    #     stream_handler.setLevel(logging.DEBUG)
    #     logger.addHandler(stream_handler)

    # logger.info(f"Start Logging, {startMessage}")
