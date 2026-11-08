"""Play MicroMagic in a browser: human vs bot, bot vs bot, and human vs human.

Run `python -m micromagic.play` from the repository root.
"""

from micromagic.play.bots import BotInfo, available_bots, make_bot
from micromagic.play.session import GameSession, new_session

__all__ = ["BotInfo", "GameSession", "available_bots", "make_bot", "new_session"]
