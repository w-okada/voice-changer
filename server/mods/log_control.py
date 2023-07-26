import logging

# logging.getLogger('numba').setLevel(logging.WARNING)


class UvicornSuppressFilter(logging.Filter):
    def filter(self, record):
        return False


def setup_loggers(startMessage: str):
    # logger = logging.getLogger("uvicorn.error")
    # logger.addFilter(UvicornSuppressFilter())

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
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('vvclient.log', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info(f"Start Logging, {startMessage}")
