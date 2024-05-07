import asyncio
import traceback

from main import setupArgParser, main
from distutils.util import strtobool
from mods.log_control import VoiceChangaerLogger

VoiceChangaerLogger.get_instance().initialize(initialize=True)
logger = VoiceChangaerLogger.get_instance().getLogger()

if __name__ == "__main__":
    parser = setupArgParser()
    parser.add_argument("--launch-browser", type=strtobool, default=True, help="Automatically launches web browser and opens the voice changer's interface.")
    args, _ = parser.parse_known_args()

    try:
        asyncio.run(main(args))
    except Exception as e:
        print(traceback.format_exc())
        input('Press Enter to continue...')
