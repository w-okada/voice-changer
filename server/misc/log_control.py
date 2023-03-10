import logging

# logging.getLogger('numba').setLevel(logging.WARNING)


class UvicornSuppressFilter(logging.Filter):
    def filter(self, record):
        return False


logger = logging.getLogger("uvicorn.error")
logger.addFilter(UvicornSuppressFilter())

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

logging.getLogger('asyncio').setLevel(logging.WARNING)
