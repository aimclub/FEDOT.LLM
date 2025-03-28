from enum import Enum

from typing_extensions import TypeAlias, Union


class BSColors(Enum):
    BLUE = "#0d6efd"
    INDIGO = "#6610f2"
    PURPLE = "#6f42c1"
    PINK = "#d63384"
    RED = "#dc3545"
    ORANGE = "#fd7e14"
    YELLOW = "#ffc107"
    GREEN = "#198754"
    TEAL = "#20c997"
    CYAN = "#0dcaf0"

    WHITE = "#fff"
    GRAY_100 = "#f8f9fa"
    GRAY_200 = "#e9ecef"
    GRAY_300 = "#dee2e6"
    GRAY_400 = "#ced4da"
    GRAY_500 = "#adb5bd"
    GRAY_600 = "#6c757d"
    GRAY_700 = "#495057"
    GRAY_800 = "#343a40"
    GRAY_900 = "#212529"
    BLACK = "#000"

    PRIMARY = BLUE
    SECONDARY = GRAY_600
    SUCCESS = GREEN
    INFO = CYAN
    WARNING = YELLOW
    DANGER = RED
    LIGHT = GRAY_100
    DARK = GRAY_800


class STColors(Enum):
    PRIMARY = "#FF4B4B"
    SECONDARY = "#262730"
    TEXT = "#FAFAFA"
    BODY = "#0E1117"


class AdditionalColors(Enum):
    TRANSPARENT = "transparent"


UIColors: TypeAlias = Union[BSColors, STColors, AdditionalColors]
