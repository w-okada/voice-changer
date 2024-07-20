import asyncio

from main import setupArgParser, main
from utils.strtobool import strtobool

import logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = setupArgParser()
    parser.add_argument("--launch-browser", type=strtobool, default=True, help="Automatically launches web browser and opens the voice changer's interface.")
    args, _ = parser.parse_known_args()

    try:
        asyncio.run(main(args))
    except Exception as e:
        logger.exception(e)
        input('Press Enter to continue...')
