"""XiangqiBench: tool-grounded xiangqi endgames for evaluating LLM agents."""

__version__ = "0.1.0"

from xiangqibench.cases import EndgameCase, load_cases
from xiangqibench.config import Config, load_config
from xiangqibench.modes import MODES, get_mode

__all__ = ["MODES", "Config", "EndgameCase", "__version__", "get_mode", "load_cases", "load_config"]
